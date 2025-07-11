#pragma once
#include "xoshiro256pp.h"

typedef struct xoshiro256pp xoshiro256pp_state;

static inline void xoshiro256pp_seed(xoshiro256pp_state *st, uint64_t seed) {
    xoshiro256pp_init(st, seed);
}

static inline uint64_t xoshiro256pp_next(xoshiro256pp_state *st) {
    uint64_t tmp[8];
    xoshiro256pp_next8(st, tmp);
    return tmp[0];
}
