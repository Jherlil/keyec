#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void blf_has_kernel(__global const ulong *bits, ulong size,
                             __global const uint *hashes,
                             __global uchar *out){
    size_t gid=get_global_id(0);
    __global const uint *h=hashes+gid*5;
    ulong a1=((ulong)h[0]<<32)|h[1];
    ulong a2=((ulong)h[2]<<32)|h[3];
    ulong a3=((ulong)h[4]<<32)|h[0];
    ulong a4=((ulong)h[1]<<32)|h[2];
    ulong a5=((ulong)h[3]<<32)|h[4];
    const uchar shifts[4]={24,28,36,40};
    int ok=1;
    for(int i=0;i<4 && ok;i++){
        uchar S=shifts[i];
        ulong idx[5]={
            (a1<<S)|(a2>>S),
            (a2<<S)|(a3>>S),
            (a3<<S)|(a4>>S),
            (a4<<S)|(a5>>S),
            (a5<<S)|(a1>>S)
        };
        for(int j=0;j<5 && ok;j++){
            ulong pos=idx[j] % (size*64UL);
            ulong word=pos/64UL;
            ulong bit=1UL<<(pos & 63UL);
            ok &= (bits[word] & bit) != 0;
        }
    }
    out[gid]=ok?1:0;
}
