// OpenCL kernel for secp256k1 scalar multiplication

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// 64bit add with carry
inline ulong addc64(ulong x, ulong y, ulong carry, __private ulong *co){
    ulong r = x + y;
    ulong c1 = r < x;
    r += carry;
    ulong c2 = r < carry;
    *co = c1 | c2;
    return r;
}

// 64bit sub with borrow
inline ulong subc64(ulong x, ulong y, ulong carry, __private ulong *bo){
    ulong r = x - y;
    ulong b1 = r > x;
    r -= carry;
    ulong b2 = r > (ulong)(-carry-1);
    *bo = b1 | b2;
    return r;
}

inline ulong umul128(ulong a, ulong b, __private ulong *hi){
    ulong lo = a * b;
    *hi = mul_hi(a,b);
    return lo;
}

typedef ulong fe[4];
typedef ulong fe320[5];

constant fe FE_P = {0xfffffffefffffc2fUL, 0xffffffffffffffffUL, 0xffffffffffffffffUL, 0xffffffffffffffffUL};

inline int fe_cmp(const fe a, const fe b){
    for(int i=3;i>=0;--i){
        if(a[i]!=b[i]) return a[i]>b[i]?1:-1;
    }
    return 0;
}

inline void fe_clone(fe r, const fe a){
    for(int i=0;i<4;i++) r[i]=a[i];
}

inline void fe_mul_scalar(fe320 r, const fe a, ulong b){
    ulong h1,h2,c=0;
    r[0]=umul128(a[0],b,&h1);
    r[1]=addc64(umul128(a[1],b,&h2),h1,c,&c);
    r[2]=addc64(umul128(a[2],b,&h1),h2,c,&c);
    r[3]=addc64(umul128(a[3],b,&h2),h1,c,&c);
    r[4]=addc64(0,h2,c,&c);
}

inline ulong fe320_addc(fe320 r,const fe320 a,const fe320 b){
    ulong c=0;
    r[0]=addc64(a[0],b[0],c,&c);
    r[1]=addc64(a[1],b[1],c,&c);
    r[2]=addc64(a[2],b[2],c,&c);
    r[3]=addc64(a[3],b[3],c,&c);
    r[4]=addc64(a[4],b[4],c,&c);
    return c;
}

inline ulong fe320_subc(fe320 r,const fe320 a,const fe320 b){
    ulong c=0;
    r[0]=subc64(a[0],b[0],c,&c);
    r[1]=subc64(a[1],b[1],c,&c);
    r[2]=subc64(a[2],b[2],c,&c);
    r[3]=subc64(a[3],b[3],c,&c);
    r[4]=subc64(a[4],b[4],c,&c);
    return c;
}

// Field operations mod P
inline void fe_modp_add(fe r,const fe a,const fe b){
    ulong c=0;
    r[0]=addc64(a[0],b[0],c,&c);
    r[1]=addc64(a[1],b[1],c,&c);
    r[2]=addc64(a[2],b[2],c,&c);
    r[3]=addc64(a[3],b[3],c,&c);
    if(c || fe_cmp(r,FE_P)>=0){
        c=0;
        r[0]=subc64(r[0],FE_P[0],c,&c);
        r[1]=subc64(r[1],FE_P[1],c,&c);
        r[2]=subc64(r[2],FE_P[2],c,&c);
        r[3]=subc64(r[3],FE_P[3],c,&c);
    }
}

inline void fe_modp_sub(fe r,const fe a,const fe b){
    ulong c=0;
    r[0]=subc64(a[0],b[0],c,&c);
    r[1]=subc64(a[1],b[1],c,&c);
    r[2]=subc64(a[2],b[2],c,&c);
    r[3]=subc64(a[3],b[3],c,&c);
    if(c){
        c=0;
        r[0]=addc64(r[0],FE_P[0],0,&c);
        r[1]=addc64(r[1],FE_P[1],c,&c);
        r[2]=addc64(r[2],FE_P[2],c,&c);
        r[3]=addc64(r[3],FE_P[3],c,&c);
    }
}

inline void fe_modp_mul(fe r,const fe a,const fe b){
    ulong rr[8]={0},tt[5]={0},c=0;
    fe_mul_scalar(rr,a,b[0]);
    fe_mul_scalar(tt,a,b[1]);
    rr[1]=addc64(rr[1],tt[0],c,&c);
    rr[2]=addc64(rr[2],tt[1],c,&c);
    rr[3]=addc64(rr[3],tt[2],c,&c);
    rr[4]=addc64(rr[4],tt[3],c,&c);
    rr[5]=addc64(rr[5],tt[4],c,&c);
    fe_mul_scalar(tt,a,b[2]);
    rr[2]=addc64(rr[2],tt[0],c,&c);
    rr[3]=addc64(rr[3],tt[1],c,&c);
    rr[4]=addc64(rr[4],tt[2],c,&c);
    rr[5]=addc64(rr[5],tt[3],c,&c);
    rr[6]=addc64(rr[6],tt[4],c,&c);
    fe_mul_scalar(tt,a,b[3]);
    rr[3]=addc64(rr[3],tt[0],c,&c);
    rr[4]=addc64(rr[4],tt[1],c,&c);
    rr[5]=addc64(rr[5],tt[2],c,&c);
    rr[6]=addc64(rr[6],tt[3],c,&c);
    rr[7]=addc64(rr[7],tt[4],c,&c);
    fe_mul_scalar(tt,rr+4,0x1000003D1UL);
    rr[0]=addc64(rr[0],tt[0],0,&c);
    rr[1]=addc64(rr[1],tt[1],c,&c);
    rr[2]=addc64(rr[2],tt[2],c,&c);
    rr[3]=addc64(rr[3],tt[3],c,&c);
    ulong hi,lo;
    lo=umul128(tt[4]+c,0x1000003D1UL,&hi);
    r[0]=addc64(rr[0],lo,0,&c);
    r[1]=addc64(rr[1],hi,c,&c);
    r[2]=addc64(rr[2],0,c,&c);
    r[3]=addc64(rr[3],0,c,&c);
    if(fe_cmp(r,FE_P)>=0) fe_modp_sub(r,r,FE_P);
}

inline void fe_modp_sqr(fe r,const fe a){
    ulong rr[8]={0},tt[5]={0},c=0,t1,t2,lo,hi;
    rr[0]=umul128(a[0],a[0],&tt[1]);
    tt[3]=umul128(a[0],a[1],&tt[4]);
    tt[3]=addc64(tt[3],tt[3],0,&c);
    tt[4]=addc64(tt[4],tt[4],c,&c);
    t1=c;
    tt[3]=addc64(tt[1],tt[3],0,&c);
    tt[4]=addc64(tt[4],0,c,&c);
    t1+=c;
    rr[1]=tt[3];
    tt[0]=umul128(a[0],a[2],&tt[1]);
    tt[0]=addc64(tt[0],tt[0],0,&c);
    tt[1]=addc64(tt[1],tt[1],c,&c);
    t2=c;
    lo=umul128(a[1],a[1],&hi);
    tt[0]=addc64(tt[0],lo,0,&c);
    tt[1]=addc64(tt[1],hi,c,&c);
    t2+=c;
    tt[0]=addc64(tt[0],tt[4],0,&c);
    tt[1]=addc64(tt[1],t1,c,&c);
    t2+=c;
    rr[2]=tt[0];
    tt[3]=umul128(a[0],a[3],&tt[4]);
    lo=umul128(a[1],a[2],&hi);
    tt[3]=addc64(tt[3],lo,0,&c);
    tt[4]=addc64(tt[4],hi,c,&c);
    t1=c+c;
    tt[3]=addc64(tt[3],tt[3],0,&c);
    tt[4]=addc64(tt[4],tt[4],c,&c);
    t1+=c;
    tt[3]=addc64(tt[1],tt[3],0,&c);
    tt[4]=addc64(tt[4],t2,c,&c);
    t1+=c;
    rr[3]=tt[3];
    tt[0]=umul128(a[1],a[3],&tt[1]);
    tt[0]=addc64(tt[0],tt[0],0,&c);
    tt[1]=addc64(tt[1],tt[1],c,&c);
    t2=c;
    lo=umul128(a[2],a[2],&hi);
    tt[0]=addc64(tt[0],lo,0,&c);
    tt[1]=addc64(tt[1],hi,c,&c);
    t2+=c;
    tt[0]=addc64(tt[0],tt[4],0,&c);
    tt[1]=addc64(tt[1],t1,c,&c);
    t2+=c;
    rr[4]=tt[0];
    tt[3]=umul128(a[2],a[3],&tt[4]);
    tt[3]=addc64(tt[3],tt[3],0,&c);
    tt[4]=addc64(tt[4],tt[4],c,&c);
    t1=c;
    tt[3]=addc64(tt[3],tt[1],0,&c);
    tt[4]=addc64(tt[4],t2,c,&c);
    t1+=c;
    rr[5]=tt[3];
    tt[0]=umul128(a[3],a[3],&tt[1]);
    tt[0]=addc64(tt[0],tt[4],0,&c);
    tt[1]=addc64(tt[1],t1,c,&c);
    rr[6]=tt[0];
    rr[7]=tt[1];
    fe_mul_scalar(tt,rr+4,0x1000003D1UL);
    rr[0]=addc64(rr[0],tt[0],0,&c);
    rr[1]=addc64(rr[1],tt[1],c,&c);
    rr[2]=addc64(rr[2],tt[2],c,&c);
    rr[3]=addc64(rr[3],tt[3],c,&c);
    lo=umul128(tt[4]+c,0x1000003D1UL,&hi);
    r[0]=addc64(rr[0],lo,0,&c);
    r[1]=addc64(rr[1],hi,c,&c);
    r[2]=addc64(rr[2],0,c,&c);
    r[3]=addc64(rr[3],0,c,&c);
    if(fe_cmp(r,FE_P)>=0) fe_modp_sub(r,r,FE_P);
}

// Elliptic curve point
typedef struct { fe x; fe y; fe z; } pe;
constant pe G1 = {
    {0x59f2815b16f81798UL,0x029bfcdb2dce28d9UL,0x55a06295ce870b07UL,0x79be667ef9dcbbacUL},
    {0x9c47d08ffb10d4b8UL,0xfd17b448a6855419UL,0x5da4fbfc0e1108a8UL,0x483ada7726a3c465UL},
    {1UL,0UL,0UL,0UL}
};

inline void pe_clone(__private pe *r, __private const pe *a){
    for(int i=0;i<4;i++){ r->x[i]=a->x[i]; r->y[i]=a->y[i]; r->z[i]=a->z[i]; }
}

inline void fe_modp_neg(fe r,const fe a){
    ulong c=0;
    r[0]=subc64(FE_P[0],a[0],c,&c);
    r[1]=subc64(FE_P[1],a[1],c,&c);
    r[2]=subc64(FE_P[2],a[2],c,&c);
    r[3]=subc64(FE_P[3],a[3],c,&c);
}

inline void _ec_jacobi_dbl1(__private pe *r,__private const pe *p){
    fe w,s,b,h,t;
    fe_modp_sqr(t,p->x);
    fe_modp_add(w,t,t);
    fe_modp_add(w,w,t);
    fe_modp_mul(s,p->y,p->z);
    fe_modp_mul(b,p->x,p->y);
    fe_modp_mul(b,b,s);
    fe_modp_add(b,b,b);
    fe_modp_add(b,b,b);
    fe_modp_add(t,b,b);
    fe_modp_sqr(h,w);
    fe_modp_sub(h,h,t);
    fe_modp_mul(r->x,h,s);
    fe_modp_add(r->x,r->x,r->x);
    fe_modp_sub(t,b,h);
    fe_modp_mul(t,w,t);
    fe_modp_sqr(r->y,p->y);
    fe_modp_sqr(h,s);
    fe_modp_mul(r->y,r->y,h);
    fe_modp_add(r->y,r->y,r->y);
    fe_modp_add(r->y,r->y,r->y);
    fe_modp_add(r->y,r->y,r->y);
    fe_modp_sub(r->y,t,r->y);
    fe_modp_mul(r->z,h,s);
    fe_modp_add(r->z,r->z,r->z);
    fe_modp_add(r->z,r->z,r->z);
    fe_modp_add(r->z,r->z,r->z);
}

inline void _ec_jacobi_add1(__private pe *r, __private const pe *p, __private const pe *q){
    fe u2,v2,u,v,w,a,vs,vc;
    fe_modp_mul(u2,p->y,q->z);
    fe_modp_mul(v2,p->x,q->z);
    fe_modp_mul(u,q->y,p->z);
    fe_modp_mul(v,q->x,p->z);
    if(! (v[0]==v2[0] && v[1]==v2[1] && v[2]==v2[2] && v[3]==v2[3]) ){
        fe_modp_mul(w,p->z,q->z);
        fe_modp_sub(u,u,u2);
        fe_modp_sub(v,v,v2);
        fe_modp_sqr(vs,v);
        fe_modp_mul(vc,vs,v);
        fe_modp_mul(vs,vs,v2);
        fe_modp_mul(r->z,vc,w);
        fe_modp_sqr(a,u);
        fe_modp_mul(a,a,w);
        fe_modp_add(w,vs,vs);
        fe_modp_sub(a,a,vc);
        fe_modp_sub(a,a,w);
        fe_modp_mul(r->x,v,a);
        fe_modp_sub(a,vs,a);
        fe_modp_mul(a,a,u);
        fe_modp_mul(u,vc,u2);
        fe_modp_sub(r->y,a,u);
    } else {
        pe_clone(r,p);
    }
}

inline void _ec_jacobi_rdc1(__private pe *r,__private const pe *a){
    fe_clone(r->z,a->z);
    fe_modp_inv(r->z,r->z);
    fe_modp_mul(r->x,a->x,r->z);
    fe_modp_mul(r->y,a->y,r->z);
    r->z[0]=1; r->z[1]=0; r->z[2]=0; r->z[3]=0;
}

inline void fe_modp_inv(fe r,const fe a){
    // binary exponentiation a^(P-2)
    fe q={1,0,0,0},p,t;
    fe_clone(p,FE_P);
    fe_clone(t,a);
    p[0]-=2;
    while(p[0]||p[1]||p[2]||p[3]){
        if(p[0]&1) fe_modp_mul(q,q,t);
        fe_modp_sqr(t,t);
        // shift right
        ulong c=0; c=p[1]<<63; p[1]=(p[1]>>1)|(p[2]<<63); p[2]=(p[2]>>1)|(p[3]<<63); p[3]>>=1; p[0]=(p[0]>>1)|c;
    }
    fe_clone(r,q);
}

inline void ec_jacobi_mul(__private pe *r,__private const pe *p,__private const fe k){
    pe t; pe_clone(&t,p);
    r->x[0]=r->y[0]=0; r->x[1]=r->y[1]=0; r->x[2]=r->y[2]=0; r->x[3]=r->y[3]=0;
    r->z[0]=1; r->z[1]=r->z[2]=r->z[3]=0;
    for(int i=0;i<256;i++){
        if(k[i/64] & (1UL<<(i%64))){
            if(r->x[0]==0 && r->y[0]==0) pe_clone(r,&t); else _ec_jacobi_add1(r,r,&t);
        }
        _ec_jacobi_dbl1(&t,&t);
    }
}

inline void ec_jacobi_mulrdc(__private pe *r,__private const pe *p,__private const fe k){
    ec_jacobi_mul(r,p,k);
    _ec_jacobi_rdc1(r,r);
}

__kernel void ecc_mul_kernel(__global ulong *out, __global const ulong *scalars){
    uint gid=get_global_id(0);
    fe k; for(int i=0;i<4;i++) k[i]=scalars[gid*4+i];
    pe res; ec_jacobi_mulrdc(&res,&G1,k);
    __global ulong *dst=out+gid*8;
    for(int i=0;i<4;i++) dst[i]=res.x[i];
    for(int i=0;i<4;i++) dst[4+i]=res.y[i];
}

