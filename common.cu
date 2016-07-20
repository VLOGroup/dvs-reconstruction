#include "common.h"
#include "common.cuh"
#include "iu/iumath.h"
#include "iu/iucore.h"


inline __device__ float sum(float3 val)
{
    return val.x+val.y+val.z;
}
inline __device__ float sum(float2 val)
{
    return val.x+val.y;
}

inline __device__ float2 abs(float2 val)
{
    return make_float2(abs(val.x),abs(val.y));
}

__global__ void set_events_kernel(iu::ImageGpu_32f_C1::KernelData output, iu::LinearDeviceMemory_32f_C4::KernelData events, float C1, float C2)
{
    int event_id = blockIdx.x*blockDim.x + threadIdx.x;

    if(event_id<events.length_) {
        float4 event = events.data_[event_id];
//        if(output(round(event.x),round(event.y))<10)
            if(event.z>0)
                output(round(event.x),round(event.y)) *= C1;
            else
                output(round(event.x),round(event.y)) /= C2;
    }
}

__global__ void set_events_kernel(iu::ImageGpu_32f_C1::KernelData output, iu::ImageGpu_32u_C1::KernelData occurence, iu::LinearDeviceMemory_32f_C4::KernelData events, float C1, float C2)
{
    int event_id = blockIdx.x*blockDim.x + threadIdx.x;

    if(event_id<events.length_) {
        float4 event = events.data_[event_id];
        const unsigned int outx = round(event.x);
        const unsigned int outy = round(event.y);
            if(event.z>0)
                output(outx,outy) *= C1;
            else
                output(outx,outy) /= C2;
        occurence(outx,outy) = 0;
    }
}

__global__ void set_timestamps_kernel(iu::ImageGpu_32f_C1::KernelData output, iu::LinearDeviceMemory_32f_C4::KernelData events)
{
    int event_id = blockIdx.x*blockDim.x + threadIdx.x;

    if(event_id<events.length_) {
        float4 event = events.data_[event_id];
        output(round(event.x),round(event.y)) = event.w;
    }
}

__global__ void upsample_kernel(iu::ImageGpu_32f_C1::KernelData output, cudaTextureObject_t tex_input, float scale)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<output.width_ && y<output.height_)
    {
        output(x,y) = tex2D<float>(tex_input,x/scale+0.5f,y/scale+0.5f);
    }
}

__global__ void upsample_exp_kernel(iu::ImageGpu_32f_C1::KernelData output, cudaTextureObject_t tex_input, float scale)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<output.width_ && y<output.height_)
    {
        output(x,y) = exp(tex2D<float>(tex_input,x/scale+0.5f,y/scale+0.5f));
    }
}
namespace cuda {
inline uint divUp(uint a, uint b) { return (a + b - 1) / b; }

void setEvents(iu::ImageGpu_32f_C1 *output,iu::ImageGpu_32f_C1 * old_timestamp, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2)
{
    iu::LinearDeviceMemory_32f_C4 events_gpu(events_host->length());
    iu::copy(events_host,&events_gpu);

    int gpu_block_x = GPU_BLOCK_SIZE*GPU_BLOCK_SIZE;
    int gpu_block_y = 1;

    // compute number of Blocks
    int nb_x = divUp(events_gpu.length(),gpu_block_x);
    int nb_y = 1;

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    set_events_kernel <<< dimGrid, dimBlock, 0 >>>(*output,events_gpu,C1,C2);
    CudaCheckError();
    set_timestamps_kernel <<< dimGrid, dimBlock, 0>>>(*old_timestamp,events_gpu);
    CudaCheckError();

//    // get last timestamp
//    float curr_time = events_host->data(events_host->length()-1)->w;
//    mv::addC(*old_timestamp,-curr_time,*lambda_time);
//    mv::mulC(*lambda_time,-1.f,*lambda_time);
}

void setEvents(iu::ImageGpu_32f_C1 *output,iu::ImageGpu_32f_C1 * old_timestamp, iu::ImageGpu_32u_C1 *occurences, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2)
{
    iu::LinearDeviceMemory_32f_C4 events_gpu(events_host->length());
    iu::copy(events_host,&events_gpu);

    int gpu_block_x = GPU_BLOCK_SIZE*GPU_BLOCK_SIZE;
    int gpu_block_y = 1;

    // compute number of Blocks
    int nb_x = divUp(events_gpu.length(),gpu_block_x);
    int nb_y = 1;

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    set_events_kernel <<< dimGrid, dimBlock, 0 >>>(*output,*occurences,events_gpu,C1,C2);
    CudaCheckError();
    set_timestamps_kernel <<< dimGrid, dimBlock, 0>>>(*old_timestamp,events_gpu);
    CudaCheckError();

}

void setEvents(iu::ImageGpu_32f_C1 *output, iu::LinearHostMemory_32f_C4 *events_host, float C1, float C2)
{
    iu::LinearDeviceMemory_32f_C4 events_gpu(events_host->length());
    iu::copy(events_host,&events_gpu);

    int gpu_block_x = GPU_BLOCK_SIZE*GPU_BLOCK_SIZE;
    int gpu_block_y = 1;

    // compute number of Blocks
    int nb_x = divUp(events_gpu.length(),gpu_block_x);
    int nb_y = 1;

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    set_events_kernel <<< dimGrid, dimBlock, 0 >>>(*output,events_gpu,C1,C2);
    CudaCheckError();
}

void upsample(iu::ImageGpu_32f_C1 *in, iu::ImageGpu_32f_C1 *out, UpsampleMethod method, bool exponentiate)
{
    int width = out->width();
    int height = out->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    in->prepareTexture(cudaReadModeElementType,method==UPSAMPLE_LINEAR? cudaFilterModeLinear:cudaFilterModePoint,cudaAddressModeClamp);
    if(exponentiate)
        upsample_exp_kernel <<<dimGrid,dimBlock, 0>>>(*out,in->getTexture(),out->width()/in->width());
    else
        upsample_kernel <<<dimGrid,dimBlock, 0>>>(*out,in->getTexture(),out->width()/in->width());
    CudaCheckError();
}

} // namespace cuda
