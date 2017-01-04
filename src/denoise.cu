// This file is part of dvs-reconstruction.
//
// Copyright (C) 2016 Christian Reinbacher <reinbacher at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// dvs-reconstruction is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// dvs-reconstruction is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#ifndef DENOISE_CU
#define DENOISE_CU

#include "denoise.h"
#include "common.h"
#include <iu/iuhelpermath.h>
#include <iu/iuio.h>
#include <iu/iumath.h>

// defines
#define GPU_BLOCK_SIZE 16

/// ugly global variables
iu::ImageGpu_32f_C2 *p=NULL;
iu::ImageGpu_32f_C1 *u_=NULL;
iu::ImageGpu_32f_C1 *ut_=NULL;
iu::ImageGpu_32f_C1 *ft=NULL;
iu::ImageGpu_32f_C2 *pt=NULL;
iu::ImageGpu_32f_C4 *manifold_t=NULL;
iu::ImageGpu_32f_C4 *p_manifold = NULL;

/// Device helper functions
inline __device__ float sqr(float a)
{
    return a*a;
}

inline __device__ float sign(float val)
{
    return (val>0)? 1.f:-1.f;
}

/// TV-L1 dual kernel for manifold denoising
__global__ void TVL1_dual_kernel(iu::ImageGpu_32f_C2::KernelData p, cudaTextureObject_t tex_u, float lambda, float sigma)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < p.width_ && y < p.height_)
    {
        const float xx = x+0.5f;
        const float yy = y+0.5f;
        float2 pc = p(x,y);
        float uc = tex2D<float>(tex_u,xx,yy);

        float2 grad = make_float2((tex2D<float>(tex_u,xx+1,yy) - uc),
                                  (tex2D<float>(tex_u,xx,yy+1) - uc));


        pc = (pc + sigma*grad );//(1+sigma*0.3f);    // dual update
//        float th = 1.f/(lambda);
        p(x,y) = pc / (max(1.f,lambda*length(pc)));
//        p(x,y) = clamp(pc,-th,th);
    }
}

/// TV-L1 primal kernel for manifold denoising
__global__ void TVL1_primal_kernel(iu::ImageGpu_32f_C1::KernelData g_u, iu::ImageGpu_32f_C1::KernelData g_u_, iu::ImageGpu_32f_C1::KernelData g_f, cudaTextureObject_t tex_p, float tau)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_u.width_ && y < g_u.height_)
    {
        const float xx = x+0.5f;
        const float yy = y+0.5f;
        float2 p = tex2D<float2>(tex_p,xx,yy);
        float f = g_f(x,y);
        float u = g_u(x,y);
        float u_ = u;

        float ktp = (p.x - tex2D<float2>(tex_p,xx-1,yy).x) +
                    (p.y - tex2D<float2>(tex_p,xx,yy-1).y);

        u = u + tau*ktp;    // primal update
        u = f+max(0.f,abs(u-f)-tau)*sign(u-f);
        g_u(x,y) = u;      // write back
        g_u_(x,y) = 2*u-u_;
    }
}

/// Precompute tensor for manifold denoising
__global__ void Prepare_manifold_kernel(iu::ImageGpu_32f_C4::KernelData g_m, cudaTextureObject_t t_t, float lambda_time)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_m.width_ && y < g_m.height_)
    {
        const float xx = x+0.5f;
        const float yy = y+0.5f;
        float t = tex2D<float>(t_t,xx,yy);
        float tx = clamp((tex2D<float>(t_t,xx+1.f,yy) - t),-0.5f,0.5f)*lambda_time;
        float ty = clamp((tex2D<float>(t_t,xx,yy+1.f) - t),-0.5f,0.5f)*lambda_time;
        float4 m;
        m.w = 1+sqr(tx)+sqr(ty);
        m.x = tx;
        m.y = ty;
        m.z = t; // maybe we need it
        g_m(x,y) = m;
    }
}

/// Precompute tensor for manifold denoising
__inline__ __device__ float3 getCoefficients(float4 mc)
{
    float c1 = (1.f+sqr(mc.y))/mc.w;
    float c3 = (1.f+sqr(mc.x))/mc.w;
    float c2 = mc.x*mc.y/mc.w;
    return make_float3(c1,c2,c3);
}

/// Dual Kernel for manifold denoising (shared between all data terms)
__global__ void TV_manifold_dual_kernel(iu::ImageGpu_32f_C4::KernelData p, iu::ImageGpu_32f_C4::KernelData m, cudaTextureObject_t tex_u, float lambda, float sigma)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < p.width_ && y < p.height_)
    {
        const float xx = x+0.5f;
        const float yy = y+0.5f;
        float4 pc = p(x,y);
        float4 mc = m(x,y);
        float uc = tex2D<float>(tex_u,xx,yy);

        float2 grad = make_float2((tex2D<float>(tex_u,xx+1,yy) - uc),
                                  (tex2D<float>(tex_u,xx,yy+1) - uc));

        float3 c = getCoefficients(mc);
        float det = sqrt(mc.w);

        float4 p_upd;
        p_upd.x = grad.x*c.x - grad.y*c.y;
        p_upd.y = grad.y*c.z - grad.x*c.y;
        p_upd.z =(mc.x*grad.x + mc.y*grad.y)/det;

        pc += sigma*p_upd;
        pc /= 1+ sigma*0.3f; // Huberize
        pc /= fmaxf(1.f,sqrtf(sqr(pc.x)+sqr(pc.y)+sqr(pc.z))/det*lambda);
        p(x,y) = pc;
    }
}

/// Convenience function to compute the divergence on the manifold
inline __device__ void computeKTP(cudaTextureObject_t tex_m, cudaTextureObject_t tex_p, unsigned int x, unsigned int y, float &ktp, float &weight)
{
    const float xx = x+0.5f;
    const float yy = y+0.5f;
    float4 p = tex2D<float4>(tex_p,xx,yy);
    float4 m_xy = tex2D<float4>(tex_m,xx,yy);
    float4 m_xm1y = tex2D<float4>(tex_m,xx-1.f,yy);
    float4 m_xym1 = tex2D<float4>(tex_m,xx,yy-1.f);
    float3 c_xy = getCoefficients(m_xy);
    float3 c_xm1y = getCoefficients(m_xm1y);
    float3 c_xym1 = getCoefficients(m_xym1);

    ktp = (p.x * (-c_xy.x + c_xy.y) +
             tex2D<float4>(tex_p,xx-1.f,yy).x * c_xm1y.x -
             tex2D<float4>(tex_p,xx,yy-1.f).x * c_xym1.y) +
            (p.y * (-c_xy.z + c_xy.y) +
             tex2D<float4>(tex_p,xx,yy-1.f).y * c_xym1.z -
             tex2D<float4>(tex_p,xx-1.f,yy).y * c_xm1y.y) +
            (p.z * (-m_xy.x/m_xy.w -m_xy.y/m_xy.w) +
             tex2D<float4>(tex_p,xx-1.f,yy).z * m_xm1y.x/m_xm1y.w +
             tex2D<float4>(tex_p,xx,yy-1.f).z * m_xym1.x/m_xym1.w);
    weight = m_xy.w;
}

/// Primal kernel for KLD data term
__global__ void TVKLD_manifold_primal_kernel(iu::ImageGpu_32f_C1::KernelData g_u, iu::ImageGpu_32f_C1::KernelData g_u_,
                                                     cudaTextureObject_t tex_p,iu::ImageGpu_32f_C1::KernelData g_f, cudaTextureObject_t tex_m, float tau, float u_min, float u_max)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_u.width_ && y < g_u.height_)
    {

        float f = g_f(x,y);
        float u = g_u(x,y);
        float u_ = u;
        float ktp,weight;

        computeKTP(tex_m,tex_p,x,y,ktp,weight);

        u = u - tau*ktp;    // primal update
        tau *= sqrtf(weight);
        u = fminf(u_max,fmaxf(u_min,(u-tau+sqrtf(sqr(u-tau)+4*tau*f))/2.f)); // KLD
        g_u(x,y) = u;      // write back
        g_u_(x,y) = 2*u-u_;
    }
}

/// Primal kernel for Log-L2 data term
__global__ void TVLogL2_manifold_primal_kernel(iu::ImageGpu_32f_C1::KernelData g_u, iu::ImageGpu_32f_C1::KernelData g_u_,
                                                     cudaTextureObject_t tex_p,iu::ImageGpu_32f_C1::KernelData g_f, cudaTextureObject_t tex_m, float tau, float u_min, float u_max)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_u.width_ && y < g_u.height_)
    {

        float f = g_f(x,y);
        float u = g_u(x,y);
        float u_ = u;
        float ktp,weight;

        computeKTP(tex_m,tex_p,x,y,ktp,weight);

        u = u - tau*ktp;    // primal update
        float v = u;
        float uu;
        tau *= sqrtf(weight);
        for(int iter=0;iter<100;iter++) { // Log L2
            float temp = log (u/f);
            float d1 = temp/u;
            float d2 = 1.f/u/u * (1.f-temp);
            if(temp>1.f) {
                d1 = 1.f/exp(1.f)/f;
                d2 = 0.f;
            }
            uu = u;
            u = min(u_max,max(u_min, u - ((u-v)+tau*d1)/ (1.f+tau*d2)));
            if (abs(uu-u) < 1e-06)
                break;
        }
        g_u(x,y) = u;      // write back
        g_u_(x,y) = 2*u-u_;
    }
}

/// Primal kernel for L2 data term
__global__ void TVL2_manifold_primal_kernel(iu::ImageGpu_32f_C1::KernelData g_u, iu::ImageGpu_32f_C1::KernelData g_u_,
                                                     cudaTextureObject_t tex_p,iu::ImageGpu_32f_C1::KernelData g_f, cudaTextureObject_t tex_m, float tau, float u_min, float u_max)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_u.width_ && y < g_u.height_)
    {

        float f = g_f(x,y);
        float u = g_u(x,y);
        float u_ = u;
        float ktp,weight;

        computeKTP(tex_m,tex_p,x,y,ktp,weight);

        u = u - tau*ktp;    // primal update
        tau *= sqrtf(weight);

        u = fminf(u_max,fmaxf(u_min,(u + tau*f) / (1+tau))); //L2
        g_u(x,y) = u;      // write back
        g_u_(x,y) = 2*u-u_;
    }
}

/// Primal kernel for L1 data term
__global__ void TVL1_manifold_primal_kernel(iu::ImageGpu_32f_C1::KernelData g_u, iu::ImageGpu_32f_C1::KernelData g_u_,
                                                     cudaTextureObject_t tex_p,iu::ImageGpu_32f_C1::KernelData g_f, cudaTextureObject_t tex_m, float tau, float u_min, float u_max)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < g_u.width_ && y < g_u.height_)
    {

        float f = g_f(x,y);
        float u = g_u(x,y);
        float u_ = u;
        float ktp,weight;

        computeKTP(tex_m,tex_p,x,y,ktp,weight);

        u = u - tau*ktp;    // primal update
        tau *= sqrtf(weight);

        u = min(u_max,max(u_min,f+max(0.f,abs(u-f)-tau)*sign(u-f))); //L1

        g_u(x,y) = u;      // write back
        g_u_(x,y) = 2*u-u_;
    }
}


namespace cuda {

void initDenoise(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1* timestamp)
{
    if (p)
        delete p;
    if (u_)
        delete u_;
    if (ut_)
        delete ut_;
    if (pt)
        delete pt;
    if(ft)
        delete ft;
    if(manifold_t)
        delete manifold_t;
    if(p_manifold)
        delete p_manifold;
    ft = new iu::ImageGpu_32f_C1(u->width(),u->height());
    p = new iu::ImageGpu_32f_C2(u->width(),u->height());
    u_ = new iu::ImageGpu_32f_C1(u->width(),u->height());
    ut_ = new iu::ImageGpu_32f_C1(u->width(),u->height());
    pt = new iu::ImageGpu_32f_C2(u->width(),u->height());
    manifold_t = new iu::ImageGpu_32f_C4(u->width(),u->height());
    p_manifold = new iu::ImageGpu_32f_C4(u->width(),u->height());

    iu::math::fill(*p,make_float2(0));
    iu::math::fill(*pt,make_float2(0));
    iu::math::fill(*p_manifold,make_float4(0));

    u_->prepareTexture(cudaReadModeElementType,cudaFilterModePoint);
    timestamp->prepareTexture(cudaReadModeElementType,cudaFilterModePoint);
    p->prepareTexture();
    ut_->prepareTexture(cudaReadModeElementType,cudaFilterModePoint);
    pt->prepareTexture();
    manifold_t->prepareTexture();
    p_manifold->prepareTexture();
}


void solveTVIncrementalManifold(iu::ImageGpu_32f_C1 *u, iu::ImageGpu_32f_C1 *f, iu::ImageGpu_32f_C1 *t, float lambda, float lambda_time, int iterations, float u_min, float u_max, METHOD method)
{
    int width = u->width();
    int height = u->height();

    int gpu_block_x = GPU_BLOCK_SIZE;
    int gpu_block_y = GPU_BLOCK_SIZE;

    // compute number of Blocks
    int nb_x = iu::divUp(width,gpu_block_x);
    int nb_y = iu::divUp(height,gpu_block_y);

    dim3 dimBlock(gpu_block_x,gpu_block_y);
    dim3 dimGrid(nb_x,nb_y);

    float L = sqrtf(8.0f);
    float tau = 1/L;
    float sigma = 1/L;

    // first denoise the time volume
    iu::copy(t,ft);
    iu::copy(t,ut_);
    for (int k = 1; k < 50; k++)
    {
        TVL1_dual_kernel<<<dimGrid, dimBlock>>>(*pt, ut_->getTexture(), 2.f, sigma);
        TVL1_primal_kernel<<<dimGrid, dimBlock>>>(*t,*ut_,*ft,pt->getTexture(),tau);
        CudaCheckError();
    }
    // debug
//    float minVal,maxVal;
//    iu::math::minMax(*t,minVal,maxVal);
//    printf ("min: %f, max %f\n",minVal,maxVal);
    // prepare manifold

    Prepare_manifold_kernel<<<dimGrid, dimBlock>>>(*manifold_t,t->getTexture(), lambda_time);
    CudaCheckError();

//    L = sqrtf(12.0f);
    L = sqrtf(8+2*sqrtf(2));
    tau = 0.01;
    sigma = 1/tau/L/L;

    // And now the intensity image
    //iu::copy(f,u);
    iu::copy(u,u_);
    for (int k = 1; k < iterations; k++)
    {
        TV_manifold_dual_kernel<<<dimGrid, dimBlock>>>(*p_manifold, *manifold_t, u_->getTexture(), lambda, sigma);
        switch(method)
        {
            case TV_L1:
                tau = 1/L;
                sigma = 1/L;
                TVL1_manifold_primal_kernel<<<dimGrid, dimBlock>>>(*u, *u_, p_manifold->getTexture(), *f, manifold_t->getTexture(), tau, u_min, u_max);
                break;
            case TV_L2:
                TVL2_manifold_primal_kernel<<<dimGrid, dimBlock>>>(*u, *u_, p_manifold->getTexture(), *f, manifold_t->getTexture(), tau, u_min, u_max);
                break;
            case TV_LogL2:
                TVLogL2_manifold_primal_kernel<<<dimGrid, dimBlock>>>(*u, *u_, p_manifold->getTexture(), *f, manifold_t->getTexture(), tau, u_min, u_max);
                break;
            case TV_KLD:
                TVKLD_manifold_primal_kernel<<<dimGrid, dimBlock>>>(*u, *u_, p_manifold->getTexture(), *f, manifold_t->getTexture(), tau, u_min, u_max);
                break;
        }

        CudaCheckError();
    }
    iu::copy(u,f);
}


} // namespace cuda

#endif
