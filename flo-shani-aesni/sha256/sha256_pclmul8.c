#include "flo-shani.h"

void sha256_pclmul8(unsigned char *message[8], unsigned int length, unsigned char *digest[8]) {
    sha256_8w(message, length, digest);
}
