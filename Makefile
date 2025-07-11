.PHONY: default clean build bench fmt add mul rnd blf remote

CC = cc
NASM = nasm
CC_FLAGS ?= -O3 -funroll-loops -fomit-frame-pointer -ffast-math -Wall -Wextra

## Optimized build flags
CFLAGS += -march=native -mavx2 -maes -msha -funroll-loops -O3 -ffast-math -pthread

ifeq ($(shell uname -m),x86_64)
CC_FLAGS += -march=x86-64-v2 -mavx2 -pthread -lpthread
endif

default: build

clean:
	@rm -rf ecloop bench main a.out *.profraw *.profdata secp256k1.o xoshiro256pp.o

build:
	$(MAKE) clean
	$(CC) $(CFLAGS) -c -I./secp256k1_fast_unsafe -I./secp256k1_fast_unsafe/include -I./secp256k1_fast_unsafe/src -include secp256k1_fast_unsafe/src/basic-config.h -DUSE_BASIC_CONFIG -DSECP256K1_BUILD secp256k1_fast_unsafe/src/secp256k1.c -o secp256k1.o
	$(CC) $(CFLAGS) -c xoshiro256pp.c -o xoshiro256pp.o
	$(CC) $(CC_FLAGS) $(CFLAGS) \
	-I./secp256k1_fast_unsafe -I./secp256k1_fast_unsafe/include \
	-I./secp256k1_fast_unsafe/src \
	-include secp256k1_fast_unsafe/src/basic-config.h \
	-DUSE_BASIC_CONFIG -DSECP256K1_BUILD \
main.c xoshiro256pp.o secp256k1.o -o ecloop

bench: build
	./ecloop bench

fmt:
	@find . -name '*.c' | xargs clang-format -i

# -----------------------------------------------------------------------------

add: build
	./ecloop add -f data/btc-puzzles-hash -r 8000:ffffff

mul: build
	cat data/btc-bw-priv | ./ecloop mul -f data/btc-bw-hash -a cu -o /dev/null

rnd: build
	./ecloop rnd -f data/btc-puzzles-hash -r 800000000000000000:ffffffffffffffff -d 0:32

blf: build
	@rm -rf /tmp/test.blf
	@printf "\n> "
	cat data/btc-bw-hash | ./ecloop blf-gen -n 32768 -o /tmp/test.blf
	@printf "\n> "
	cat data/btc-bw-hash | ./ecloop blf-gen -n 32768 -o /tmp/test.blf
	@printf "\n> "
	./ecloop add -f /tmp/test.blf -r 8000:ffffff -q -o /dev/null
	@printf "\n> "
	cat data/btc-bw-priv | ./ecloop mul -f /tmp/test.blf -a cu -o /dev/null

verify: build
	./ecloop mult-verify
# -----------------------------------------------------------------------------
# https://btcpuzzle.info/puzzle
	
range_28 = 8000000:fffffff
range_32 = 80000000:ffffffff
range_33 = 100000000:1ffffffff
range_34 = 200000000:3ffffffff
range_35 = 400000000:7ffffffff
range_36 = 800000000:fffffffff
range_71 = 400000000000000000:7fffffffffffffffff
range_72 = 800000000000000000:ffffffffffffffff
range_73 = 1000000000000000000:1ffffffffffffffffff
range_74 = 2000000000000000000:3ffffffffffffffffff
range_76 = 8000000000000000000:fffffffffffffffffff
range_77 = 10000000000000000000:1fffffffffffffffffff
range_78 = 20000000000000000000:3fffffffffffffffffff
range_79 = 40000000000000000000:7fffffffffffffffffff
_RANGES_ = $(foreach r,$(filter range_%,$(.VARIABLES)),$(patsubst range_%,%,$r))
	
puzzle: build
	@$(if $(filter $(_RANGES_),$(n)),$(error "Invalid range $(n)"))
	./ecloop rnd -f data/btc-puzzles-hash -d 0:32 -r $(range_$(n)) -o ./found_$(n).txt

%:
	@$(if $(filter $(_RANGES_),$@),make --no-print-directory puzzle n=$@,)

# -----------------------------------------------------------------------------

host=mele
cmd=add

remote:
	@rsync -arc --progress --delete-after --exclude={'ecloop','found*.txt','.git'} ./ $(host):/tmp/ecloop
	@ssh -tt $(host) 'clear; $(CC) --version'
	ssh -tt $(host) 'cd /tmp/ecloop; make $(cmd) CC=$(CC)'

	bench-compare:
	@ssh -tt $(host) " \
	cd /tmp; rm -rf ecloop keyhunt; \
	cd /tmp && git clone https://github.com/vladkens/ecloop.git && cd ecloop && make CC=clang; \
	echo '--------------------------------------------------'; \
	cd /tmp && git clone https://github.com/albertobsd/keyhunt.git && cd keyhunt && make; \
	echo '--------------------------------------------------'; \
	cd /tmp; \
	echo '--- t=1 (keyhunt)'; \
time ./keyhunt/keyhunt -m rmd160 -f ecloop/data/btc-bw-hash -r 8000:fffffff -t 1 -n 16777216; \
echo '--- t=1 (ecloop)'; \
time ./ecloop/ecloop add -f ecloop/data/btc-bw-hash -t 1 -r 8000:fffffff; \
echo '--- t=4 (keyhunt)'; \
time ./keyhunt/keyhunt -m rmd160 -f ecloop/data/btc-bw-hash -r 8000:fffffff -t 4 -n 16777216; \
echo '--- t=4 (ecloop)'; \
time ./ecloop/ecloop add -f ecloop/data/btc-bw-hash -t 4 -r 8000:fffffff; \
"
