/**
  * @file	CLUtils.h
  * @brief	Header file with declarations OpenCL utilites.
  *
  * @author Sergey Zavalishin <s.zavalishin@samsung.com>
  * 
  * Copyright (c) Sergey Zavalishin 2012-2015
  */

#ifndef _CLUTILS_H_
#define _CLUTILS_H_

#ifdef __cplusplus
extern "C"
{
#endif 

#include <CL/cl.h>

#define CL_DEVICE_NAME_SIZE 256
#define CL_KERNEL_FILE_NAME_SIZE 256
#define CL_BUILD_PARAMS_STRING_SIZE 256

//CL device info
typedef struct clDeficeInfo
{
    cl_device_id	device_ID;
    int				num_cores;	
    char			device_name[CL_DEVICE_NAME_SIZE];
    cl_uint			min_align;
}
clDeviceInfo;

//OpenCL context
typedef struct clState
{
    cl_context			context;
    cl_command_queue	queue;
    cl_program			program;
    clDeviceInfo		device_info;
} 
clState;

//OpenCL target platform
typedef enum CL_PLATFORM
{
    DEFAULT = 0, AMD, NVIDIA, INTEL
}
CL_PLATFORM;

//OpenCL init params
typedef struct clInitParams
{
    cl_device_type  device_type;
    char            build_params[CL_BUILD_PARAMS_STRING_SIZE];
    char            kernel_source_file_name[CL_KERNEL_FILE_NAME_SIZE];
}
clInitParams;

int InitOpenCL( clState* state, clInitParams* params );
int TerminateOpenCL( clState* state );



/**
  * @brief      Inits new kernel
  * @param		[out]	kernel			Pointer to cl_kernel
  * @param		[in]	context			Pointer to CL context structure
  * @param		[in]	kernel_name		Pointer to kernel name string
  * @return		                        0 if successful, non-0 on error
  */
int clInitKernel(cl_kernel *kernel, const clState *context, const char *kernel_name);

/**
  * @brief      Executes kernel within current context 
  * @param		[in]	context			Pointer to CL context structure
  * @param		[in]	params			Pointer to kernel params array
  * @param		[in]	param_count		Number of params
  * @param		[in]	work_size		Number of elements to be processed
  * @param		[in]	work_size		Number of elements in local work-group (or 0)
  * @return		                        0 if successful, non-0 on error
  */
int clExecuteKernel(const clState *context, const cl_kernel* kernel, const void **kernel_params, const size_t *kernel_params_sizes,
					int param_count, size_t global_work_size, size_t local_work_size, int dim_count);




#ifdef __cplusplus
}
#endif 

#endif  /* _CLUTILS_H_ */
