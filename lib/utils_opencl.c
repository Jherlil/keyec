#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;
typedef unsigned int u32;
typedef u64 h160_t[5];

typedef struct blf_t {
    size_t size;
    u64 *bits;
} blf_t;

static cl_program build_program(cl_context ctx, cl_device_id dev){
    FILE *f=fopen("lib/utils_opencl.cl","r");
    if(!f){ fprintf(stderr,"Failed to open utils_opencl.cl\n"); return NULL; }
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    char *src=malloc(sz+1); fread(src,1,sz,f); src[sz]='\0'; fclose(f);
    const char *sources[]={src};
    cl_int err; cl_program p=clCreateProgramWithSource(ctx,1,sources,NULL,&err);
    if(err!=CL_SUCCESS){ free(src); return NULL; }
    err=clBuildProgram(p,1,&dev,"",NULL,NULL);
    if(err!=CL_SUCCESS){
        size_t log_size; clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&log_size);
        char *log=malloc(log_size+1);
        clGetProgramBuildInfo(p,dev,CL_PROGRAM_BUILD_LOG,log_size,log,NULL);
        log[log_size]='\0'; fprintf(stderr,"OpenCL build error:\n%s\n",log);
        free(log); clReleaseProgram(p); p=NULL;
    }
    free(src); return p;
}

static int blf_has_batch(uint8_t *out, blf_t *blf, const h160_t *hashes, size_t count){
    cl_int err; cl_platform_id platform; cl_device_id device;
    err=clGetPlatformIDs(1,&platform,NULL); if(err!=CL_SUCCESS) return 0;
    err=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL); if(err!=CL_SUCCESS) return 0;
    cl_context ctx=clCreateContext(NULL,1,&device,NULL,NULL,&err); if(err!=CL_SUCCESS) return 0;
    cl_command_queue q=clCreateCommandQueue(ctx,device,0,&err); if(err!=CL_SUCCESS){ clReleaseContext(ctx); return 0; }
    cl_program prog=build_program(ctx,device); if(!prog){ clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }
    cl_kernel kern=clCreateKernel(prog,"blf_has_kernel",&err); if(err!=CL_SUCCESS){ clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }
    size_t bits_size=blf->size*sizeof(u64);
    size_t hash_size=count*5*sizeof(u32);
    size_t out_size=count*sizeof(uint8_t);
    cl_mem buf_bits=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bits_size,blf->bits,&err); if(err!=CL_SUCCESS) goto cleanup;
    cl_mem buf_hash=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,hash_size,(void*)hashes,&err); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_bits); goto cleanup; }
    cl_mem buf_out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out_size,NULL,&err); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_bits); clReleaseMemObject(buf_hash); goto cleanup; }
    cl_ulong size=blf->size;
    clSetKernelArg(kern,0,sizeof(buf_bits),&buf_bits);
    clSetKernelArg(kern,1,sizeof(size),&size);
    clSetKernelArg(kern,2,sizeof(buf_hash),&buf_hash);
    clSetKernelArg(kern,3,sizeof(buf_out),&buf_out);
    size_t global=count; err=clEnqueueNDRangeKernel(q,kern,1,NULL,&global,NULL,0,NULL,NULL);
    if(err!=CL_SUCCESS){ clReleaseMemObject(buf_bits); clReleaseMemObject(buf_hash); clReleaseMemObject(buf_out); goto cleanup; }
    clFinish(q);
    err=clEnqueueReadBuffer(q,buf_out,CL_TRUE,0,out_size,out,0,NULL,NULL);
    clReleaseMemObject(buf_bits); clReleaseMemObject(buf_hash); clReleaseMemObject(buf_out);
cleanup:
    clReleaseKernel(kern); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return err==CL_SUCCESS;
}

int blf_has_opencl_batch(uint8_t *out, blf_t *blf, const h160_t *hashes, size_t count){
    return blf_has_batch(out, blf, hashes, count);
}

