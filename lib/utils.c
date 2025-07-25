// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.

#pragma once

#include "ecc.c"
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "xoshiro256ss.h"

#ifdef _WIN32
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <termios.h>
#endif

typedef char hex40[41]; // rmd160 hex string
typedef char hex64[65]; // sha256 hex string
typedef u32 h160_t[5];

// Mark: Terminal

#define COLOR_YELLOW "\033[33m"
#define COLOR_RESET "\033[0m"

void term_clear_line() {
  fprintf(stderr, "\033[2K\r");
  // in case if ecloop will be piped
  fflush(stdout);
  fflush(stderr);
}

// MARK: helpers

u64 tsnow() {
  struct timespec ts;
  // clock_gettime(CLOCK_MONOTONIC, &ts);
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000 + ts.tv_nsec / 1e6;
}

bool strendswith(const char *str, const char *suffix) {
  size_t str_len = strlen(str);
  size_t suffix_len = strlen(suffix);
  return (str_len >= suffix_len) && (strcmp(str + str_len - suffix_len, suffix) == 0);
}

char *strtrim(char *str) {
  if (str == NULL) return NULL;

  char *since = str;
  while (isspace((unsigned char)*since)) ++since;

  char *until = str + strlen(str) - 1;
  while (until > since && isspace((unsigned char)*until)) --until;

  *(until + 1) = '\0';
  if (since != until) memmove(str, since, until - since + 2);

  return str;
}

// MARK: random helpers

static FILE *_urandom = NULL;

static void _close_urandom(void) {
  if (_urandom != NULL) {
    fclose(_urandom);
    _urandom = NULL;
  }
}

// xoshiro256** PRNG state
static struct xoshiro256ss xrng __attribute__((aligned(64)));
#define XRNG_BUF_SIZE 4096
static u64 xrng_buf[XRNG_BUF_SIZE] __attribute__((aligned(64)));
static size_t xrng_idx = XRNG_BUF_SIZE;

void prng_seed(u64 seed) {
  xoshiro256ss_init(&xrng, seed);
  xrng_idx = XRNG_BUF_SIZE;
}

INLINE u64 prng_next64() {
  if (xrng_idx >= XRNG_BUF_SIZE) {
    xoshiro256ss_filln(&xrng, (uint64_t *)xrng_buf,
                       XRNG_BUF_SIZE / XOSHIRO256SS_WIDTH);
    xrng_idx = 0;
  }
  return xrng_buf[xrng_idx++];
}

void prng_next8(u64 out[8]) {
  if (xrng_idx + 8 > XRNG_BUF_SIZE) {
    xoshiro256ss_filln(&xrng, (uint64_t *)xrng_buf,
                       XRNG_BUF_SIZE / XOSHIRO256SS_WIDTH);
    xrng_idx = 0;
  }
  memcpy(out, &xrng_buf[xrng_idx], 8 * sizeof(u64));
  xrng_idx += 8;
}

u64 _prand64() { return prng_next64(); }

u64 _urand64() {
  if (_urandom == NULL) {
    _urandom = fopen("/dev/urandom", "rb");
    if (_urandom == NULL) {
      fprintf(stderr, "failed to open /dev/urandom\n");
      exit(1);
    }

    atexit(_close_urandom);
  }

  u64 r;
  if (fread(&r, sizeof(r), 1, _urandom) != 1) {
    fprintf(stderr, "failed to read from /dev/urandom\n");
    exit(1);
  }

  return r;
}

INLINE u64 rand64(bool urandom) { return urandom ? _urand64() : _prand64(); }

u32 encode_seed(const char *seed) {
  u32 hash = 0;
  while (*seed) {
    char c = *seed++;
    hash = (hash << 5) - hash + (unsigned char)c;
    hash &= 0xFFFFFFFF;
  }
  return hash;
}

// MARK: fe_random

void fe_prand(fe r) {
  __attribute__((aligned(64))) u64 buf[8];
  prng_next8(buf);
  for (int i = 0; i < 4; ++i) r[i] = buf[i];
  r[3] &= 0xfffffffefffffc2f;
}

void fe_urand(fe r) {
  for (int i = 0; i < 4; ++i) r[i] = _urand64();
  r[3] &= 0xfffffffefffffc2f;
}

void fe_rand_range(fe r, const fe a, const fe b, bool urandom) {
  fe range, x;
  fe_modn_sub(range, b, a); // range = b - a
  fe_add64(range, 1);       // range = range + 1

  size_t bits = fe_bitlen(range);
  assert(bits > 0 && bits <= 256);

  do {
    urandom ? fe_urand(x) : fe_prand(x);

    // drop unused bits
    int top = (bits - 1) / 64;
    for (int i = top + 1; i < 4; ++i) x[i] = 0;

    int rem = bits % 64;
    if (rem) x[top] &= (1ULL << rem) - 1;

  } while (fe_cmp(x, range) >= 0);

  fe_modn_add(x, x, a);
  assert(fe_cmp(x, a) >= 0);
  assert(fe_cmp(x, b) <= 0);
  fe_clone(r, x);
}

// MARK: args

typedef struct args_t {
  int argc;
  const char **argv;
} args_t;

bool args_bool(args_t *args, const char *name) {
  for (int i = 1; i < args->argc; ++i) {
    if (strcmp(args->argv[i], name) == 0) return true;
  }
  return false;
}

u64 args_uint(args_t *args, const char *name, int def) {
  for (int i = 1; i < args->argc - 1; ++i) {
    if (strcmp(args->argv[i], name) == 0) {
      return strtoull(args->argv[i + 1], NULL, 10);
    }
  }
  return def;
}

char *arg_str(args_t *args, const char *name) {
  for (int i = 1; i < args->argc; ++i) {
    if (strcmp(args->argv[i], name) == 0) {
      if (i + 1 < args->argc) return (char *)args->argv[i + 1];
    }
  }
  return NULL;
}

// MARK: queue

typedef struct queue_item_t {
  void *data_ptr;
  struct queue_item_t *next;
} queue_item_t;

typedef struct queue_t {
  size_t capacity;
  size_t size;
  bool done;
  queue_item_t *head;
  queue_item_t *tail;
  pthread_mutex_t lock;
  pthread_cond_t cond_put;
  pthread_cond_t cond_get;
} queue_t;

void queue_init(queue_t *q, size_t capacity) {
  q->capacity = capacity;
  q->size = 0;
  q->done = false;
  q->head = NULL;
  q->tail = NULL;
  pthread_mutex_init(&q->lock, NULL);
  pthread_cond_init(&q->cond_put, NULL);
  pthread_cond_init(&q->cond_get, NULL);
}

void queue_done(queue_t *q) {
  pthread_mutex_lock(&q->lock);
  q->done = true;
  pthread_cond_broadcast(&q->cond_get);
  pthread_mutex_unlock(&q->lock);
}

void queue_put(queue_t *q, void *data_ptr) {
  pthread_mutex_lock(&q->lock);
  if (q->done) {
    pthread_mutex_unlock(&q->lock);
    return;
  }

  while (q->size == q->capacity) {
    pthread_cond_wait(&q->cond_put, &q->lock);
  }

  queue_item_t *item = malloc(sizeof(queue_item_t));
  item->data_ptr = data_ptr;
  item->next = NULL;

  if (q->tail != NULL) q->tail->next = item;
  else q->head = item;

  q->tail = item;
  q->size += 1;

  pthread_cond_signal(&q->cond_get);
  pthread_mutex_unlock(&q->lock);
}

void *queue_get(queue_t *q) {
  pthread_mutex_lock(&q->lock);
  while (q->size == 0 && !q->done) {
    pthread_cond_wait(&q->cond_get, &q->lock);
  }

  if (q->size == 0) {
    pthread_mutex_unlock(&q->lock);
    return NULL;
  }

  queue_item_t *item = q->head;
  q->head = item->next;
  if (!q->head) q->tail = NULL;

  void *data_ptr = item->data_ptr;
  free(item);
  q->size -= 1;

  pthread_cond_signal(&q->cond_put);
  pthread_mutex_unlock(&q->lock);
  return data_ptr;
}

// MARK: bloom filter

#define BLF_MAGIC 0x45434246 // FourCC: ECBF
#define BLF_VERSION 1

typedef struct blf_t {
  size_t size;
  u64 *bits;
} blf_t;

static inline void blf_setbit(blf_t *blf, size_t idx) {
  blf->bits[idx % (blf->size * 64) / 64] |= (u64)1 << (idx % 64);
}

static inline bool blf_getbit(blf_t *blf, u64 idx) {
  return (blf->bits[idx % (blf->size * 64) / 64] & ((u64)1 << (idx % 64))) != 0;
}

void blf_add(blf_t *blf, const h160_t hash) {
  u64 a1 = (u64)hash[0] << 32 | hash[1];
  u64 a2 = (u64)hash[2] << 32 | hash[3];
  u64 a3 = (u64)hash[4] << 32 | hash[0];
  u64 a4 = (u64)hash[1] << 32 | hash[2];
  u64 a5 = (u64)hash[3] << 32 | hash[4];

  u8 shifts[4] = {24, 28, 36, 40};
  for (size_t i = 0; i < 4; ++i) {
    u8 S = shifts[i];
    blf_setbit(blf, a1 << S | a2 >> S);
    blf_setbit(blf, a2 << S | a3 >> S);
    blf_setbit(blf, a3 << S | a4 >> S);
    blf_setbit(blf, a4 << S | a5 >> S);
    blf_setbit(blf, a5 << S | a1 >> S);
  }
}

bool blf_has(blf_t *blf, const h160_t hash) {
  u64 a1 = (u64)hash[0] << 32 | hash[1];
  u64 a2 = (u64)hash[2] << 32 | hash[3];
  u64 a3 = (u64)hash[4] << 32 | hash[0];
  u64 a4 = (u64)hash[1] << 32 | hash[2];
  u64 a5 = (u64)hash[3] << 32 | hash[4];

  u8 shifts[4] = {24, 28, 36, 40};
  for (size_t i = 0; i < 4; ++i) {
    u8 S = shifts[i];
    if (!blf_getbit(blf, a1 << S | a2 >> S)) return false;
    if (!blf_getbit(blf, a2 << S | a3 >> S)) return false;
    if (!blf_getbit(blf, a3 << S | a4 >> S)) return false;
    if (!blf_getbit(blf, a4 << S | a5 >> S)) return false;
    if (!blf_getbit(blf, a5 << S | a1 >> S)) return false;
  }

  return true;
}

#ifdef __AVX2__
#include <immintrin.h>
// vectorized bloom filter check for 4 hashes
void blf_has4(uint8_t out[4], blf_t *blf, const h160_t *hashes) {
  __m256i a1 = _mm256_set_epi64x((u64)hashes[3][0] << 32 | hashes[3][1],
                                 (u64)hashes[2][0] << 32 | hashes[2][1],
                                 (u64)hashes[1][0] << 32 | hashes[1][1],
                                 (u64)hashes[0][0] << 32 | hashes[0][1]);
  __m256i a2 = _mm256_set_epi64x((u64)hashes[3][2] << 32 | hashes[3][3],
                                 (u64)hashes[2][2] << 32 | hashes[2][3],
                                 (u64)hashes[1][2] << 32 | hashes[1][3],
                                 (u64)hashes[0][2] << 32 | hashes[0][3]);
  __m256i a3 = _mm256_set_epi64x((u64)hashes[3][4] << 32 | hashes[3][0],
                                 (u64)hashes[2][4] << 32 | hashes[2][0],
                                 (u64)hashes[1][4] << 32 | hashes[1][0],
                                 (u64)hashes[0][4] << 32 | hashes[0][0]);
  __m256i a4 = _mm256_set_epi64x((u64)hashes[3][1] << 32 | hashes[3][2],
                                 (u64)hashes[2][1] << 32 | hashes[2][2],
                                 (u64)hashes[1][1] << 32 | hashes[1][2],
                                 (u64)hashes[0][1] << 32 | hashes[0][2]);
  __m256i a5 = _mm256_set_epi64x((u64)hashes[3][3] << 32 | hashes[3][4],
                                 (u64)hashes[2][3] << 32 | hashes[2][4],
                                 (u64)hashes[1][3] << 32 | hashes[1][4],
                                 (u64)hashes[0][3] << 32 | hashes[0][4]);

  const int shifts[4] = {24, 28, 36, 40};
  for (int i = 0; i < 4; ++i) out[i] = 1;

  for (int s = 0; s < 4; ++s) {
    int S = shifts[s];
    __m256i i1 = _mm256_or_si256(_mm256_slli_epi64(a1, S), _mm256_srli_epi64(a2, S));
    __m256i i2 = _mm256_or_si256(_mm256_slli_epi64(a2, S), _mm256_srli_epi64(a3, S));
    __m256i i3 = _mm256_or_si256(_mm256_slli_epi64(a3, S), _mm256_srli_epi64(a4, S));
    __m256i i4 = _mm256_or_si256(_mm256_slli_epi64(a4, S), _mm256_srli_epi64(a5, S));
    __m256i i5 = _mm256_or_si256(_mm256_slli_epi64(a5, S), _mm256_srli_epi64(a1, S));

    alignas(32) u64 idx[5][4];
    _mm256_store_si256((__m256i *)idx[0], i1);
    _mm256_store_si256((__m256i *)idx[1], i2);
    _mm256_store_si256((__m256i *)idx[2], i3);
    _mm256_store_si256((__m256i *)idx[3], i4);
    _mm256_store_si256((__m256i *)idx[4], i5);

    for (int lane = 0; lane < 4; ++lane) {
      if (!out[lane]) continue;
      if (!blf_getbit(blf, idx[0][lane]) || !blf_getbit(blf, idx[1][lane]) ||
          !blf_getbit(blf, idx[2][lane]) || !blf_getbit(blf, idx[3][lane]) ||
          !blf_getbit(blf, idx[4][lane]))
        out[lane] = 0;
    }
  }
}
#else
void blf_has4(uint8_t out[4], blf_t *blf, const h160_t *hashes) {
  for (int i = 0; i < 4; ++i) out[i] = blf_has(blf, hashes[i]);
}
#endif

void blf_has8(uint8_t out[8], blf_t *blf, const h160_t *hashes) {
#ifdef __AVX2__
  blf_has4(out, blf, hashes);
  blf_has4(out + 4, blf, hashes + 4);
#else
  for (int i = 0; i < 8; ++i) out[i] = blf_has(blf, hashes[i]);
#endif
}

bool blf_save(const char *filepath, blf_t *blf) {
  FILE *file = fopen(filepath, "wb");
  if (file == NULL) {
    fprintf(stderr, "failed to open output file\n");
    exit(1);
  }

  u32 blf_magic = BLF_MAGIC;
  u32 blg_version = BLF_VERSION;

  if (fwrite(&blf_magic, sizeof(blf_magic), 1, file) != 1) {
    fprintf(stderr, "failed to write bloom filter magic\n");
    return false;
  };

  if (fwrite(&blg_version, sizeof(blg_version), 1, file) != 1) {
    fprintf(stderr, "failed to write bloom filter version\n");
    return false;
  }

  if (fwrite(&blf->size, sizeof(blf->size), 1, file) != 1) {
    fprintf(stderr, "failed to write bloom filter size\n");
    return false;
  }

  if (fwrite(blf->bits, sizeof(u64), blf->size, file) != blf->size) {
    fprintf(stderr, "failed to write bloom filter bits\n");
    return false;
  }

  fclose(file);
  return true;
}

bool blf_load(const char *filepath, blf_t *blf) {
  FILE *file = fopen(filepath, "rb");
  if (file == NULL) {
    fprintf(stderr, "failed to open input file\n");
    return false;
  }

  u32 blf_magic, blf_version;
  size_t size;

  bool is_ok = true;
  is_ok = is_ok && fread(&blf_magic, sizeof(blf_magic), 1, file) == 1;
  is_ok = is_ok && fread(&blf_version, sizeof(blf_version), 1, file) == 1;
  is_ok = is_ok && fread(&size, sizeof(size), 1, file) == 1;
  if (!is_ok) {
    fprintf(stderr, "failed to read bloom filter header\n");
    return false;
  }

  if (blf_magic != BLF_MAGIC || blf_version != BLF_VERSION) {
    fprintf(stderr, "invalid bloom filter version; create a new filter with blf-gen command\n");
    return false;
  }

  u64 *bits = calloc(size, sizeof(u64));
  if (fread(bits, sizeof(u64), size, file) != size) {
    fprintf(stderr, "failed to read bloom filter bits\n");
    return false;
  }

  fclose(file);
  blf->size = size;
  blf->bits = bits;
  return true;
}

// MARK: blf-gen command

void __blf_gen_usage(args_t *args) {
  printf("Usage: %s blf-gen -n <count> -o <file>\n", args->argv[0]);
  printf("Generate a bloom filter from a list of hex-encoded hash160 values passed to stdin.\n");
  printf("\nOptions:\n");
  printf("  -n <count>      - Number of hashes to add.\n");
  printf("  -o <file>       - File to write bloom filter (must have a .blf extension).\n");
  exit(1);
}

void blf_gen(args_t *args) {
  u64 n = args_uint(args, "-n", 0);
  if (n == 0) {
    fprintf(stderr, "[!] missing filter size (-n <number>)\n");
    return __blf_gen_usage(args);
  }

  char *filepath = arg_str(args, "-o");
  if (filepath == NULL) {
    fprintf(stderr, "[!] missing output file (-o <file>)\n");
    return __blf_gen_usage(args);
  }

  // https://hur.st/bloomfilter/?n=500M&p=1e9&m=&k=20
  u64 r = 1e9;
  double p = 1.0 / (double)r;
  u64 m = (u64)(n * log(p) / log(1.0 / pow(2.0, log(2.0))));
  double mb = (double)m / 8 / 1024 / 1024;
  size_t size = (m + 63) / 64;

  blf_t blf = {.size = 0, .bits = NULL};
  if (access(filepath, F_OK) == 0) {
    char *todo = "delete it or choose a different file";
    printf("file %s already exists; loading...\n", filepath);

    if (!blf_load(filepath, &blf)) {
      fprintf(stderr, "[!] failed to load bloom filter: %s\n", todo);
      exit(1);
    }

    if (blf.size != size) {
      fprintf(stderr, "[!] bloom filter size mismatch (%'zu != %'zu): %s\n", blf.size, size, todo);
      exit(1);
    }

    printf("updating bloom filter...\n");
  } else {
    printf("creating bloom filter...\n");
    blf.size = size;
    blf.bits = calloc(blf.size, sizeof(u64));
  }

  printf("bloom filter params: n = %'llu | p = 1:%'llu | m = %'llu (%'.1f MB)\n", n, r, m, mb);

  u64 count = 0;
  hex40 line;
  while (fgets(line, sizeof(line), stdin) != NULL) {
    if (strlen(line) != sizeof(line) - 1) continue;

    h160_t hash;
    for (size_t j = 0; j < sizeof(line) - 1; j += 8) sscanf(line + j, "%8x", &hash[j / 8]);

    if (blf_has(&blf, hash)) continue;

    blf_add(&blf, hash);
    count += 1;
  }

  printf("added %'llu new items; saving to %s\n", count, filepath);

  if (!blf_save(filepath, &blf)) {
    fprintf(stderr, "[!] failed to save bloom filter\n");
    exit(1);
  }

  free(blf.bits);
}

// MARK: blf-check command

void __blf_check_usage(args_t *args) {
  printf("Usage: %s blf-check -f <file> <hash> [hash...]\n", args->argv[0]);
  printf("Check if one or more hex-encoded hash160 values are in the bloom filter.\n");
  printf("\nOptions:\n");
  printf("  -f <file>       Path to the bloom filter file (required).\n");
  printf("\nArguments:\n");
  printf("  <hash>          One or more hex-encoded hash160 values to check.\n");
  printf("                  If no arguments are provided, stdin will be used as source.\n");
  exit(1);
}

bool __blf_check_hex(blf_t *blf, const char *hex) {
  h160_t h = {0};
  for (size_t i = 0; i < 40; i += 8) sscanf(hex + i, "%8x", &h[i / 8]);
  return blf_has(blf, h);
}

void blf_check(args_t *args) {
  char *filepath = arg_str(args, "-f");
  if (filepath == NULL) {
    fprintf(stderr, "[!] missing input file (-f <file>)\n");
    return __blf_check_usage(args);
  }

  blf_t blf = {.size = 0, .bits = NULL};
  if (!blf_load(filepath, &blf)) {
    fprintf(stderr, "[!] failed to load bloom filter\n");
    exit(1);
  }

  bool has_opts = false;
  for (int i = 1; i < args->argc; ++i) {
    if (strlen(args->argv[i]) != 40) continue;

    has_opts = true;
    bool found = __blf_check_hex(&blf, args->argv[i]);
    printf("%s %s\n", args->argv[i], found ? "FOUND" : "NOT FOUND");
  }

  if (has_opts) return;

  char line[128];
  while (fgets(line, sizeof(line), stdin) != NULL) {
    strtrim(line);
    // printf("checking %s (%'zu)...\n", line, strlen(line));
    if (strlen(line) != 40) continue; // 40 hex chars + \n

    bool found = __blf_check_hex(&blf, line);
    printf("%s %s\n", line, found ? "FOUND" : "NOT FOUND");
  }
}

// Mark: CPU count

int get_cpu_count() {
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return (int)sysinfo.dwNumberOfProcessors;
#else
  int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
  return MAX(1, cpu_count);
#endif
}

// MARK: TTY

typedef void (*tty_cb_t)(void *ctx, const char ch);

typedef struct {
  tty_cb_t cb;
  void *ctx;
} tty_thread_args_t;

#ifdef _WIN32

void tty_cleanup() {}
void tty_init(tty_cb_t cb, void *ctx) { atexit(tty_cleanup); }

#else

struct termios _orig_termios;
int _tty_fd = -1;

void *_tty_listener(void *arg) {
  tty_thread_args_t *args = (tty_thread_args_t *)arg;
  tty_cb_t cb = args->cb;
  void *ctx = args->ctx;
  free(args);

  fd_set fds;
  char ch;

  while (true) {
    if (_tty_fd < 0) break;

    // todo: race condition with tty_cleanup
    int tty_fd = _tty_fd;

    FD_ZERO(&fds);
    FD_SET(tty_fd, &fds);

    int ret = select(tty_fd + 1, &fds, NULL, NULL, NULL);
    if (ret < 0) break;

    if (FD_ISSET(tty_fd, &fds)) {
      if (read(tty_fd, &ch, 1) > 0) {
        if (cb) cb(ctx, ch);
      }
    }
  }

  return NULL;
}

void tty_cleanup() {
  if (_tty_fd < 0) return;

  tcsetattr(_tty_fd, TCSANOW, &_orig_termios);
  close(_tty_fd);
  _tty_fd = -1;
}

void tty_init(tty_cb_t cb, void *ctx) {
  atexit(tty_cleanup);

  _tty_fd = open("/dev/tty", O_RDONLY | O_NONBLOCK);
  if (_tty_fd < 0) return;

  tcgetattr(_tty_fd, &_orig_termios);

  struct termios raw = _orig_termios;
  raw.c_lflag &= ~(ICANON | ECHO); // disable canonical mode and echo
  tcsetattr(_tty_fd, TCSANOW, &raw);

  tty_thread_args_t *args = malloc(sizeof(tty_thread_args_t));
  if (!args) return;
  args->cb = cb;
  args->ctx = ctx;

  // thread will exit when _tty_fd is closed
  pthread_t _tty_thread = 0;
  pthread_create(&_tty_thread, NULL, _tty_listener, args);
}

#endif
