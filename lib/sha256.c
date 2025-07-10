#pragma once
#include "compat.c"
#include "../flo-shani-aesni/sha256/flo-shani.h"

static inline void sha256_final(u32 state[8], const u8 data[], u32 length) {
    unsigned char digest[32];
    sha256_update_shani(data, (unsigned long)length, digest);
    for (int i = 0; i < 8; ++i) {
        state[i] = ((u32)digest[i*4] << 24) |
                   ((u32)digest[i*4+1] << 16) |
                   ((u32)digest[i*4+2] << 8) |
                   (u32)digest[i*4+3];
    }
}
