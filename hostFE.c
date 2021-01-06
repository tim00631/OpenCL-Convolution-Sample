#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

#define CL_KERNEL_NAME "convolution"
void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    cl_kernel kernel = clCreateKernel(*program, CL_KERNEL_NAME, &status);
    CHECK(status, "clCreateKernel");
    cl_mem inputImageBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * imageHeight * imageWidth, inputImage, &status);
    CHECK(status, "clCreateBuffer for input");
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * filterSize, filter, &status);
    CHECK(status, "clCreateBuffer for filter");
    cl_mem outputImageBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * imageHeight * imageWidth, NULL, &status);
    CHECK(status, "clCreateBuffer for output");
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImageBuffer);
    status |=clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterBuffer);
    status |=clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputImageBuffer);
    status |=clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth);
    status |=clSetKernelArg(kernel, 4, sizeof(cl_int), &imageHeight);
    status |=clSetKernelArg(kernel, 5, sizeof(cl_int), &filterWidth);
    CHECK(status, "clSetKernelArg");

    const size_t global_work_size [2] = {imageWidth, imageHeight};
    const size_t local_work_size [2] = {16 ,16};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRangeKernel");
    status = clEnqueueReadBuffer(queue, outputImageBuffer, CL_TRUE, 0, sizeof(float)* imageWidth * imageHeight, outputImage, 0, NULL, NULL);

    // Release memory
    clReleaseCommandQueue(queue);
}
