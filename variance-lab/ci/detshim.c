// LD_PRELOAD shim that removes sources of run-to-run non-determinism from a
// benchmarked process: wall/monotonic clocks, kernel entropy, thread-pool size.
//
// Features are selected with DETSHIM=<comma list>:
//   clock   clock_gettime/gettimeofday/time return a virtual clock that starts at
//           a fixed epoch and advances by DETSHIM_STEP_NS (default 1000) per call.
//   random  getrandom/getentropy fill buffers from a fixed-seed xorshift stream;
//           reads from /dev/urandom and /dev/random do the same.
//   threads pthread_create succeeds only DETSHIM_MAX_THREADS times per process
//           (default 1) and then fails with EAGAIN. tokio's blocking pool treats
//           EAGAIN as temporary and runs the queued task on an existing worker,
//           so the pool is pinned to that many threads.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#define FIXED_REALTIME_SEC 1792540800LL  /* 2026-10-21T00:00:00Z, after the frozen fixtures */
#define FIXED_MONOTONIC_SEC 1000LL
#define DEFAULT_STEP_NS 1000LL

static int feature_clock = -1;
static int feature_random = -1;
static int feature_threads = -1;
static long long step_ns = DEFAULT_STEP_NS;
static long max_threads = 1;
static _Atomic long threads_created = 0;
static _Atomic long long ticks = 0;
static _Atomic uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static _Atomic int urandom_fd = -1;

static void init_features(void) {
    if (feature_clock >= 0) return;
    const char *spec = getenv("DETSHIM");
    int clock_on = 0, random_on = 0, threads_on = 0;
    if (spec) {
        clock_on = strstr(spec, "clock") != NULL;
        random_on = strstr(spec, "random") != NULL;
        threads_on = strstr(spec, "threads") != NULL;
    }
    const char *step = getenv("DETSHIM_STEP_NS");
    if (step && *step) step_ns = atoll(step);
    const char *cap = getenv("DETSHIM_MAX_THREADS");
    if (cap && *cap) max_threads = atol(cap);
    feature_random = random_on;
    feature_threads = threads_on;
    feature_clock = clock_on;
}

static long long next_virtual_ns(void) {
    long long n = atomic_fetch_add(&ticks, 1) + 1;
    return n * step_ns;
}

static void fill_deterministic(void *buf, size_t len) {
    unsigned char *out = buf;
    for (size_t i = 0; i < len; i++) {
        uint64_t x = atomic_load(&rng_state);
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        atomic_store(&rng_state, x);
        out[i] = (unsigned char)(x >> 24);
    }
}

int clock_gettime(clockid_t clk, struct timespec *ts) {
    init_features();
    if (!feature_clock) {
        static int (*real)(clockid_t, struct timespec *) = NULL;
        if (!real) real = dlsym(RTLD_NEXT, "clock_gettime");
        return real(clk, ts);
    }
    long long ns = next_virtual_ns();
    long long base = (clk == CLOCK_REALTIME || clk == CLOCK_REALTIME_COARSE || clk == CLOCK_TAI)
        ? FIXED_REALTIME_SEC : FIXED_MONOTONIC_SEC;
    ts->tv_sec = base + ns / 1000000000LL;
    ts->tv_nsec = ns % 1000000000LL;
    return 0;
}

int gettimeofday(struct timeval *tv, void *tz) {
    init_features();
    if (!feature_clock) {
        static int (*real)(struct timeval *, void *) = NULL;
        if (!real) real = dlsym(RTLD_NEXT, "gettimeofday");
        return real(tv, tz);
    }
    if (tv) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        tv->tv_sec = ts.tv_sec;
        tv->tv_usec = ts.tv_nsec / 1000;
    }
    return 0;
}

time_t time(time_t *out) {
    init_features();
    if (!feature_clock) {
        static time_t (*real)(time_t *) = NULL;
        if (!real) real = dlsym(RTLD_NEXT, "time");
        return real(out);
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    if (out) *out = ts.tv_sec;
    return ts.tv_sec;
}

ssize_t getrandom(void *buf, size_t len, unsigned int flags) {
    init_features();
    if (!feature_random) {
        static ssize_t (*real)(void *, size_t, unsigned int) = NULL;
        if (!real) real = dlsym(RTLD_NEXT, "getrandom");
        return real(buf, len, flags);
    }
    fill_deterministic(buf, len);
    return (ssize_t)len;
}

int getentropy(void *buf, size_t len) {
    init_features();
    if (!feature_random) {
        static int (*real)(void *, size_t) = NULL;
        if (!real) real = dlsym(RTLD_NEXT, "getentropy");
        return real(buf, len);
    }
    if (len > 256) { errno = EIO; return -1; }
    fill_deterministic(buf, len);
    return 0;
}

static int is_random_device(const char *path) {
    return path && (strcmp(path, "/dev/urandom") == 0 || strcmp(path, "/dev/random") == 0);
}

int open(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open");
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    int fd = real(path, flags, mode);
    init_features();
    if (feature_random && fd >= 0 && is_random_device(path)) atomic_store(&urandom_fd, fd);
    return fd;
}

int open64(const char *path, int flags, ...) {
    static int (*real)(const char *, int, ...) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "open64");
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) {
        va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap);
    }
    int fd = real(path, flags, mode);
    init_features();
    if (feature_random && fd >= 0 && is_random_device(path)) atomic_store(&urandom_fd, fd);
    return fd;
}

ssize_t read(int fd, void *buf, size_t len) {
    static ssize_t (*real)(int, void *, size_t) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "read");
    init_features();
    if (feature_random && fd >= 0 && fd == atomic_load(&urandom_fd)) {
        fill_deterministic(buf, len);
        return (ssize_t)len;
    }
    return real(fd, buf, len);
}

int close(int fd) {
    static int (*real)(int) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "close");
    if (fd >= 0 && fd == atomic_load(&urandom_fd)) atomic_store(&urandom_fd, -1);
    return real(fd);
}

// Only threads requesting the Rust std default stack (2 MiB, set via
// DETSHIM_CAP_STACK_SIZE) are counted against the cap; pools that choose their
// own stack size (uv's rayon pool and CLI runtime use 4 MiB) are left alone.
static int is_capped_thread(const pthread_attr_t *attr) {
    size_t size = 0;
    if (!attr || pthread_attr_getstacksize(attr, &size) != 0) return 0;
    const char *spec = getenv("DETSHIM_CAP_STACK_SIZE");
    size_t capped = spec && *spec ? (size_t)atoll(spec) : (size_t)2 * 1024 * 1024;
    return size == capped;
}

int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start)(void *), void *arg) {
    static int (*real)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *) = NULL;
    if (!real) real = dlsym(RTLD_NEXT, "pthread_create");
    init_features();
    int capped = feature_threads && is_capped_thread(attr);
    if (capped && atomic_load(&threads_created) >= max_threads) return EAGAIN;
    int rc = real(thread, attr, start, arg);
    if (rc == 0 && capped) atomic_fetch_add(&threads_created, 1);
    return rc;
}
