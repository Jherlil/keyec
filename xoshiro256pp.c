#include "xoshiro256pp.h"

static uint64_t splitmix64(uint64_t *st) {
    uint64_t z = (*st += UINT64_C(0x9E3779B97f4A7C15));
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static inline __m256i rotl256(__m256i x, int k) {
    return _mm256_or_si256(_mm256_slli_epi64(x, k),
                           _mm256_srli_epi64(x, 64 - k));
}

static void scalar_next(uint64_t s[4]) {
    uint64_t t = s[1] << 17;
    uint64_t result = ((s[0] + s[3]) << 23 | (s[0] + s[3]) >> (64 - 23)) + s[0];
    (void)result;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 45) | (s[3] >> (64 - 45));
}

static void scalar_jump128(uint64_t s[4]) {
    static const uint64_t JUMP[4] = { 0x180ec6d33cfd0abaULL,
                                      0xd5a61266f0c9392cULL,
                                      0xa9582618e03fc9aaULL,
                                      0x39abdc4529b1661cULL };
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (size_t i = 0; i < 4; i++)
        for (size_t b = 0; b < 64; b++) {
            if (JUMP[i] & (1ULL << b)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            scalar_next(s);
        }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

int xoshiro256pp_init(struct xoshiro256pp *rng, uint64_t seed) {
    uint64_t smx = seed;
    uint64_t st[4];
    for (int j = 0; j < 4; ++j) st[j] = splitmix64(&smx);
    for (int lane = 0; lane < 8; ++lane) {
        scalar_jump128(st);
        rng->s0[lane] = st[0];
        rng->s1[lane] = st[1];
        rng->s2[lane] = st[2];
        rng->s3[lane] = st[3];
    }
    return 1;
}

void xoshiro256pp_next8(struct xoshiro256pp *rng, uint64_t out[8]) {
    for (int half = 0; half < 2; ++half) {
        __m256i s0 = _mm256_load_si256((__m256i *)(rng->s0 + half * 4));
        __m256i s1 = _mm256_load_si256((__m256i *)(rng->s1 + half * 4));
        __m256i s2 = _mm256_load_si256((__m256i *)(rng->s2 + half * 4));
        __m256i s3 = _mm256_load_si256((__m256i *)(rng->s3 + half * 4));

        __m256i result = rotl256(_mm256_add_epi64(s0, s3), 23);
        result = _mm256_add_epi64(result, s0);
        __m256i t = _mm256_slli_epi64(s1, 17);

        s2 = _mm256_xor_si256(s2, s0);
        s3 = _mm256_xor_si256(s3, s1);
        s1 = _mm256_xor_si256(s1, s2);
        s0 = _mm256_xor_si256(s0, s3);
        s2 = _mm256_xor_si256(s2, t);
        s3 = rotl256(s3, 45);

        _mm256_store_si256((__m256i *)(rng->s0 + half * 4), s0);
        _mm256_store_si256((__m256i *)(rng->s1 + half * 4), s1);
        _mm256_store_si256((__m256i *)(rng->s2 + half * 4), s2);
        _mm256_store_si256((__m256i *)(rng->s3 + half * 4), s3);

        _mm256_store_si256((__m256i *)(out + half * 4), result);
    }
}

