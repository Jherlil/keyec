__constant uint SHA256_K[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

__constant uint SHA256_IV[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

__constant uint _n[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};

__constant uchar _r[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};

__constant uint n_[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

__constant uchar r_[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

#define F1(x,y,z) ((x) ^ (y) ^ (z))
#define F2(x,y,z) (((x)&(y)) | (~(x)&(z)))
#define F3(x,y,z) (((x)|~(y)) ^ (z))
#define F4(x,y,z) (((x)&(z)) | ((y)&~(z)))
#define F5(x,y,z) ((x) ^ ((y)|~(z)))

#define ROTL32(x,n) rotate((x),(n))
#define ROTR32(x,n) rotate((x),(32-(n)))

inline void store64be(ulong v,__private uchar*out){
    out[0]=(uchar)(v>>56); out[1]=(uchar)(v>>48); out[2]=(uchar)(v>>40); out[3]=(uchar)(v>>32);
    out[4]=(uchar)(v>>24); out[5]=(uchar)(v>>16); out[6]=(uchar)(v>>8); out[7]=(uchar)v;
}

inline void sha256_init(__private uint S[8]){
    for(int i=0;i<8;i++) S[i]=SHA256_IV[i];
}

inline void sha256_process_block(__private uint S[8], __private const uchar block[64]){
    __private uint w[64];
    for(int i=0;i<16;i++){
        int k=i*4;
        w[i]=((uint)block[k]<<24)|((uint)block[k+1]<<16)|((uint)block[k+2]<<8)|block[k+3];
    }
    for(int i=16;i<64;i++){
        uint x=w[i-15];
        uint y=w[i-2];
        uint s0=ROTR32(x,7)^ROTR32(x,18)^(x>>3);
        uint s1=ROTR32(y,17)^ROTR32(y,19)^(y>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint a=S[0],b=S[1],c=S[2],d=S[3],e=S[4],f=S[5],g=S[6],h=S[7];
    for(int i=0;i<64;i++){
        uint t1=h+(ROTR32(e,6)^ROTR32(e,11)^ROTR32(e,25))+((e&f)^((~e)&g))+SHA256_K[i]+w[i];
        uint t2=(ROTR32(a,2)^ROTR32(a,13)^ROTR32(a,22))+((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    S[0]+=a; S[1]+=b; S[2]+=c; S[3]+=d; S[4]+=e; S[5]+=f; S[6]+=g; S[7]+=h;
}

inline void sha256(const __private uchar *msg,int blocks,__private uint digest[8]){
    sha256_init(digest);
    for(int i=0;i<blocks;i++) sha256_process_block(digest,msg+i*64);
}

inline void rmd160_process(__private uint state[5], __private const uint x[16]){
    uint a1,b1,c1,d1,e1,a2,b2,c2,d2,e2,alpha,beta;
    a1=a2=0x67452301u; b1=b2=0xefcdab89u; c1=c2=0x98badcfeu; d1=d2=0x10325476u; e1=e2=0xc3d2e1f0u;
    for(int i=0;i<16;i++){ alpha=a1+F1(b1,c1,d1)+x[_n[i]]; alpha=ROTL32(alpha,_r[i])+e1; beta=ROTL32(c1,10); a1=e1; c1=b1; e1=d1; b1=alpha; d1=beta; alpha=a2+F5(b2,c2,d2)+x[n_[i]]+0x50a28be6u; alpha=ROTL32(alpha,r_[i])+e2; beta=ROTL32(c2,10); a2=e2; c2=b2; e2=d2; b2=alpha; d2=beta; }
    for(int i=16;i<32;i++){ alpha=a1+F2(b1,c1,d1)+x[_n[i]]+0x5a827999u; alpha=ROTL32(alpha,_r[i])+e1; beta=ROTL32(c1,10); a1=e1; c1=b1; e1=d1; b1=alpha; d1=beta; alpha=a2+F4(b2,c2,d2)+x[n_[i]]+0x5c4dd124u; alpha=ROTL32(alpha,r_[i])+e2; beta=ROTL32(c2,10); a2=e2; c2=b2; e2=d2; b2=alpha; d2=beta; }
    for(int i=32;i<48;i++){ alpha=a1+F3(b1,c1,d1)+x[_n[i]]+0x6ed9eba1u; alpha=ROTL32(alpha,_r[i])+e1; beta=ROTL32(c1,10); a1=e1; c1=b1; e1=d1; b1=alpha; d1=beta; alpha=a2+F3(b2,c2,d2)+x[n_[i]]+0x6d703ef3u; alpha=ROTL32(alpha,r_[i])+e2; beta=ROTL32(c2,10); a2=e2; c2=b2; e2=d2; b2=alpha; d2=beta; }
    for(int i=48;i<64;i++){ alpha=a1+F4(b1,c1,d1)+x[_n[i]]+0x8f1bbcdcu; alpha=ROTL32(alpha,_r[i])+e1; beta=ROTL32(c1,10); a1=e1; c1=b1; e1=d1; b1=alpha; d1=beta; alpha=a2+F2(b2,c2,d2)+x[n_[i]]+0x7a6d76e9u; alpha=ROTL32(alpha,r_[i])+e2; beta=ROTL32(c2,10); a2=e2; c2=b2; e2=d2; b2=alpha; d2=beta; }
    for(int i=64;i<80;i++){ alpha=a1+F5(b1,c1,d1)+x[_n[i]]+0xa953fd4eu; alpha=ROTL32(alpha,_r[i])+e1; beta=ROTL32(c1,10); a1=e1; c1=b1; e1=d1; b1=alpha; d1=beta; alpha=a2+F1(b2,c2,d2)+x[n_[i]]; alpha=ROTL32(alpha,r_[i])+e2; beta=ROTL32(c2,10); a2=e2; c2=b2; e2=d2; b2=alpha; d2=beta; }
    d2+=c1+0xefcdab89u; state[1]=0x98badcfeu+d1+e2; state[2]=0x10325476u+e1+a2; state[3]=0xc3d2e1f0u+a1+b2; state[4]=0x67452301u+b1+c2; state[0]=d2;
    for(int i=0;i<5;i++) state[i]=as_uint(as_uchar4(state[i]).s3210);
}

__kernel void addr33_kernel(__global const ulong *pts, __global uint *out){
    uint gid=get_global_id(0);
    const ulong *p=pts+gid*8;
    ulong x3=p[3],x2=p[2],x1=p[1],x0=p[0];
    ulong y0=p[4];
    uchar msg[64]={0};
    msg[0]=(y0&1)?0x03:0x02;
    store64be(x3,msg+1);
    store64be(x2,msg+9);
    store64be(x1,msg+17);
    store64be(x0,msg+25);
    msg[33]=0x80; msg[62]=0x01; msg[63]=0x08;
    uint sha[8];
    sha256(msg,1,sha);
    uint block[16]={0};
    for(int i=0;i<8;i++) block[i]=as_uint(as_uchar4(sha[i]).s3210);
    block[8]=0x80000000u; block[14]=256u;
    uint rmd[5];
    rmd160_process(rmd,block);
    __global uint *dst=out+gid*5;
    for(int i=0;i<5;i++) dst[i]=rmd[i];
}

__kernel void addr65_kernel(__global const ulong *pts, __global uint *out){
    uint gid=get_global_id(0);
    const ulong *p=pts+gid*8;
    ulong x3=p[3],x2=p[2],x1=p[1],x0=p[0];
    ulong y3=p[7],y2=p[6],y1=p[5],y0=p[4];
    uchar msg[128]={0};
    msg[0]=0x04;
    store64be(x3,msg+1); store64be(x2,msg+9); store64be(x1,msg+17); store64be(x0,msg+25);
    store64be(y3,msg+33); store64be(y2,msg+41); store64be(y1,msg+49); store64be(y0,msg+57);
    msg[65]=0x80; msg[126]=0x02; msg[127]=0x08;
    uint sha[8];
    sha256(msg,2,sha);
    uint block[16]={0};
    for(int i=0;i<8;i++) block[i]=as_uint(as_uchar4(sha[i]).s3210);
    block[8]=0x80000000u; block[14]=256u;
    uint rmd[5];
    rmd160_process(rmd,block);
    __global uint *dst=out+gid*5;
    for(int i=0;i<5;i++) dst[i]=rmd[i];
}
