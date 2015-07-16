/**
  * @file	CLutils.c
  * @brief	Implementation of OpenCL utilites.
  */

#include "CLUtils.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

static char *ReadSource(const char *fileName)
{
    FILE *file = fopen(fileName, "rb");
	long size = 0;
	char *src = NULL;
	size_t res = 0;

    if (!file)
    {
        return NULL;
    }

    if (fseek(file, 0, SEEK_END))
    {
		fclose(file);
        return NULL;
    }

    size = ftell(file);
    if (size == 0)
    {
		fclose(file);
        return NULL;
    }

    rewind(file);

    src = (char *)calloc(size + 1, sizeof(char));
    if (!src)
    {
		fclose(file);
        return NULL;
    }

    res = fread(src, 1, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
		fclose(file);
		free(src);
        return NULL;
    }

    src[size] = '\0'; /* NULL terminated */
    fclose(file);

    return src;
}

int TerminateOpenCL( clState* state )
{
    if( state == NULL )
    {
        return 0;
    }

    if( state->program )
    {
        clReleaseProgram(state->program);
        state->program = NULL;
    }

    if( state->queue )
    {
        clReleaseCommandQueue(state->queue);
        state->queue = NULL;
    }

    if( state->context )
    {
        clReleaseContext(state->context);
        state->context = NULL;
    }

    return 0;
};

int InitOpenCL( clState* state, clInitParams* params )
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id* platformIDs;
    cl_platform_id platformID;
    cl_device_id deviceID = (cl_device_id)0;
    char* sourceCode = NULL;
    int notFound = 1;
    cl_uint i;

    if(!params || !state)
    {
        return 1; //no params specified
    }

    memset(state, 0, sizeof(*state));

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if( errNum != CL_SUCCESS )
    {
        return 1;
    }

    platformIDs = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    if( platformIDs == NULL )
    {
        return 1;
    }

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    if( errNum != CL_SUCCESS )
    {
        return 1;
    }

    // Iterate through the list of platforms until we find one that supports
    // device that we want, otherwise fail with an error.
    for (i = 0; i < numPlatforms; i++)
    {
        errNum = clGetDeviceIDs(platformIDs[i], params->device_type, 0, NULL, &numDevices);

        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        {
            return 1;
        }
        else if (numDevices > 0) 
        {
            errNum = clGetDeviceIDs(platformIDs[i], params->device_type, 1, &deviceID, NULL);
            platformID = platformIDs[i];
			
            if(errNum != CL_SUCCESS)
            {
                continue;
            }
            notFound = 0;

            break;
       }
    }

    free(platformIDs);

    // Check to see if we found at least one device, otherwise return
    if ( notFound == 1 ) 
    {
        TerminateOpenCL(state);
        return 1;
    }

    memset(state, 0, sizeof(state));

    //create context
    state->context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, NULL);
    if(state->context == NULL)
    {
        TerminateOpenCL(state);
        return 2; //can't create context
    }

    //create command queue
    state->queue = clCreateCommandQueue(state->context, deviceID, 0, NULL);
    if(state->queue == NULL)
    {
        TerminateOpenCL(state);
        return 3; //can't create command queue
    }

    //create program with source
    sourceCode = ReadSource(params->kernel_source_file_name);
    if(!sourceCode)
    {
        TerminateOpenCL(state);
        return 4; //can't find kernel source file
    }
    state->program = clCreateProgramWithSource(state->context, 1, (const char**)&sourceCode, NULL, NULL);
    if(state->program == NULL)
    {
        TerminateOpenCL(state);
        free((void*)sourceCode);
        return 5; //can't create create program with source
    }

    //build program
    errNum = clBuildProgram(state->program, 0, NULL, params->build_params, NULL, NULL);

    if(errNum != CL_SUCCESS)
    {
        size_t len = 0;
        char *buffer;
        clGetProgramBuildInfo(state->program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        buffer = malloc(len * sizeof(char));
        clGetProgramBuildInfo(state->program, deviceID, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
        printf("%s\n", buffer);

        free(buffer);
        TerminateOpenCL(state);
        free((void*)sourceCode);

        return 6; //can't build program
    }

    // Setting up device info
    state->device_info.device_ID = deviceID;

    errNum = clGetDeviceInfo(deviceID, CL_DEVICE_NAME, CL_DEVICE_NAME_SIZE,
    		(void *)&state->device_info.device_name, NULL);
	
	errNum |= clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
		&state->device_info.num_cores, NULL);

	errNum |= clGetDeviceInfo(deviceID, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint),
		&state->device_info.min_align, NULL);

	if(errNum != CL_SUCCESS)
	{
		TerminateOpenCL(state);
		return 7; // Can't get device info
	}

	state->device_info.min_align /= 8; //min memory align in bytes

    return 0; // Everything is ok
}


int clInitKernel(cl_kernel *kernel, const clState *context, const char *kernel_name)
{
    cl_int errCode;

	//check params
	if(!kernel || !context || !kernel_name)
		return 1; //wrong input params

	//creating kernel
	if(!context->program)
		return 1; //no program specified

	*kernel = clCreateKernel(context->program, kernel_name, &errCode);
    if(errCode != CL_SUCCESS)
    {
        return 3; //can't create kernel
    }
	return 0; //everything is ok
};

static int clTerminateKernel(cl_kernel *kernel)
{
    return clReleaseKernel(*kernel); 
};

int clExecuteKernel(const clState *context, const cl_kernel* kernel, const void **kernel_params, const size_t *kernel_params_sizes,
					int param_count, size_t global_work_size, size_t local_work_size, int dim_count){
	cl_int err = 0;
	cl_uint i = 0;
	
	//check params
	if(!context || param_count < 0 || dim_count < 1)
		return 1; //wrong params

	if(!kernel || !context->queue)
		return 2; //kernel is not specified or no queue

	//set kernel args
	if(kernel_params)
		for(i = 0; i < param_count; i++){
			err = clSetKernelArg(*kernel, i, kernel_params_sizes[i], kernel_params[i]);
			if(err != CL_SUCCESS)
				return 3; //can't set kernel arg
		}

	//start kernel execution
	err = clEnqueueNDRangeKernel(context->queue, *kernel, dim_count, NULL, &global_work_size, 
		local_work_size > 0 ? &local_work_size : NULL, 0, NULL, NULL);

	if(err != CL_SUCCESS)
		return 4; //can't execute kernel

	return 0; //everything is ok
};