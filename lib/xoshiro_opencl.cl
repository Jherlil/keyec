__kernel void xoshiro_kernel(__global ulong *state, uint rounds, __global ulong *out){
    uint lane = get_global_id(0);
    __global ulong *s0p = state + lane;
    __global ulong *s1p = state + lane + 8;
    __global ulong *s2p = state + lane + 16;
    __global ulong *s3p = state + lane + 24;
    ulong s0=*s0p, s1=*s1p, s2=*s2p, s3=*s3p;
    for(uint i=0;i<rounds;i++){
        ulong result = rotate(s1*5UL,7)*9UL;
        ulong t = s1<<17;
        s2 ^= s0;
        s3 ^= s1;
        s1 ^= s2;
        s0 ^= s3;
        s2 ^= t;
        s3 = rotate(s3,45);
        out[i*8+lane]=result;
    }
    *s0p=s0; *s1p=s1; *s2p=s2; *s3p=s3;
}
