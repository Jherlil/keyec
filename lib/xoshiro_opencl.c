#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xoshiro256ss.h"

static cl_program build_program(cl_context ctx, cl_device_id dev){
    FILE *f=fopen("lib/xoshiro_opencl.cl","r");
    if(!f){ fprintf(stderr,"Failed to open xoshiro_opencl.cl\n"); return NULL; }
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

int xoshiro256ss_opencl_filln(struct xoshiro256ss *rng, uint64_t *buf, size_t n){
    cl_int err; cl_platform_id platform; cl_device_id device;
    err=clGetPlatformIDs(1,&platform,NULL); if(err!=CL_SUCCESS) return 0;
    err=clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&device,NULL); if(err!=CL_SUCCESS) return 0;
    cl_context ctx=clCreateContext(NULL,1,&device,NULL,NULL,&err); if(err!=CL_SUCCESS) return 0;
    cl_command_queue q=clCreateCommandQueue(ctx,device,0,&err); if(err!=CL_SUCCESS){ clReleaseContext(ctx); return 0; }
    cl_program prog=build_program(ctx,device); if(!prog){ clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }
    cl_kernel kern=clCreateKernel(prog,"xoshiro_kernel",&err); if(err!=CL_SUCCESS){ clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }

    size_t state_size=4*XOSHIRO256SS_WIDTH*sizeof(uint64_t);
    size_t out_size=n*XOSHIRO256SS_WIDTH*sizeof(uint64_t);
    cl_mem buf_state=clCreateBuffer(ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,state_size,rng->s,&err); if(err!=CL_SUCCESS) goto cleanup;
    cl_mem buf_out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,out_size,NULL,&err); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_state); goto cleanup; }

    clSetKernelArg(kern,0,sizeof(buf_state),&buf_state);
    cl_uint rounds=(cl_uint)n; clSetKernelArg(kern,1,sizeof(rounds),&rounds);
    clSetKernelArg(kern,2,sizeof(buf_out),&buf_out);
    size_t global=XOSHIRO256SS_WIDTH;
    err=clEnqueueNDRangeKernel(q,kern,1,NULL,&global,NULL,0,NULL,NULL); if(err!=CL_SUCCESS){ clReleaseMemObject(buf_state); clReleaseMemObject(buf_out); goto cleanup; }
    clFinish(q);
    err=clEnqueueReadBuffer(q,buf_out,CL_TRUE,0,out_size,buf,0,NULL,NULL);
    if(err==CL_SUCCESS)
        clEnqueueReadBuffer(q,buf_state,CL_TRUE,0,state_size,rng->s,0,NULL,NULL);
    clReleaseMemObject(buf_state); clReleaseMemObject(buf_out);
cleanup:
    clReleaseKernel(kern); clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return err==CL_SUCCESS;
}
