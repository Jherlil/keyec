#ifndef XOSHIRO256PP_H
#define XOSHIRO256PP_H

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

struct xoshiro256pp {
    uint64_t s0[8];
    uint64_t s1[8];
    uint64_t s2[8];
    uint64_t s3[8];
};

int xoshiro256pp_init(struct xoshiro256pp *rng, uint64_t seed);
void xoshiro256pp_next8(struct xoshiro256pp *rng, uint64_t out[8]);

#endif /* XOSHIRO256PP_H */
