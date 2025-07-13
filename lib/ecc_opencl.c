#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;

typedef u64 fe[4];
typedef struct { fe x; fe y; } pe;

static cl_program build_program(cl_context ctx, cl_device_id dev){
    FILE *f=fopen("lib/ecc_opencl.cl","r");
    if(!f){ fprintf(stderr,"Failed to open ecc_opencl.cl\n"); return NULL; }
    fseek(f,0,SEEK_END); long size=ftell(f); rewind(f);
    char *src=malloc(size+1); fread(src,1,size,f); src[size]='\0'; fclose(f);
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

int ecc_mul_opencl_batch(pe *out,const fe *scalars,size_t count){
    cl_int err; cl_platform_id platform; cl_device_id device;
    err=clGetPlatformIDs(1,&platform,NULL); if(err!=CL_SUCCESS) return 0;
    err=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL); if(err!=CL_SUCCESS) return 0;
    cl_context ctx=clCreateContext(NULL,1,&device,NULL,NULL,&err); if(err!=CL_SUCCESS) return 0;
    cl_command_queue q=clCreateCommandQueue(ctx,device,0,&err);
    if(err!=CL_SUCCESS){ clReleaseContext(ctx); return 0; }
    cl_program prog=build_program(ctx,device); if(!prog){ clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }
    cl_kernel kern=clCreateKernel(prog,"ecc_mul_kernel",&err); if(err!=CL_SUCCESS){ clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }
    size_t in_size=count*4*sizeof(u64); size_t out_size=count*8*sizeof(u64);
    cl_mem buf_in=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,in_size,(void*)scalars,&err); if(err!=CL_SUCCESS) goto cleanup;
    cl_mem buf_out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out_size,NULL,&err); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_in); goto cleanup; }
    clSetKernelArg(kern,0,sizeof(buf_out),&buf_out);
    clSetKernelArg(kern,1,sizeof(buf_in),&buf_in);
    size_t global=count; err=clEnqueueNDRangeKernel(q,kern,1,NULL,&global,NULL,0,NULL,NULL); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_in); clReleaseMemObject(buf_out); goto cleanup; }
    clFinish(q);
    err=clEnqueueReadBuffer(q,buf_out,CL_TRUE,0,out_size,out,0,NULL,NULL);
    clReleaseMemObject(buf_in); clReleaseMemObject(buf_out);
cleanup:
    clReleaseKernel(kern); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return err==CL_SUCCESS;
}
