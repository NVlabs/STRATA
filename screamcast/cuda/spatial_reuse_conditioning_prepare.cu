// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Cirrus-42 (AI agent): share native replicate padding and AMP conversion.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

// Fuse align_corners=False interpolation, replicate padding and BF16 layout.
// Native generated interpolation uses horizontal lerps then a vertical lerp.
__global__ void prepare_reuse(const float* x, __nv_bfloat16* a, __nv_bfloat16* b) {
    __shared__ float coarse[2][32][33];
    __shared__ __nv_bfloat16 transpose[32][529];
    int lane=threadIdx.x%32, warp=threadIdx.x/32, plane=blockIdx.z;
    int y0=max(0,min(31,(int)blockIdx.x-1));
    int y1=min(y0+1,31);
    #pragma unroll
    for(int j=0;j<4;++j) {
        int c=blockIdx.y*32+warp+j*8;
        const float* src=x+(int64_t)(plane/24)*25165824+c*24576+(plane%24)*1024;
        coarse[0][warp+j*8][lane]=src[y0*32+lane];
        coarse[1][warp+j*8][lane]=src[y1*32+lane];
    }
    __syncthreads();
    float wy[4];
    #pragma unroll
    for(int r=0;r<4;++r) {
        int h=max(0,min(127,(int)blockIdx.x*4+r-2));
        float fy=fmaxf((h+0.5f)*0.25f-0.5f,0.0f);
        wy[r]=fy-(int)fy;
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        int local_c=warp+j*8;
        #pragma unroll
        for(int k=0;k<5;++k) {
            int w=k*32+lane;
            if(w<132) {
                int pixel_w=max(0,min(127,w-2));
                float fx=fmaxf((pixel_w+0.5f)*0.25f-0.5f,0.0f);
                int x0=(int)fx, x1=min(x0+1,31);
                float wx=fx-x0;
                float v00=coarse[0][local_c][x0], v01=coarse[0][local_c][x1];
                float v10=coarse[1][local_c][x0], v11=coarse[1][local_c][x1];
                float top=fmaf(v01-v00,wx,v00), bottom=fmaf(v11-v10,wx,v10);
                float delta=bottom-top;
                #pragma unroll
                for(int r=0;r<4;++r)
                    transpose[local_c][r*132+w]=__float2bfloat16_rn(fmaf(delta,wy[r],top));
            }
        }
    }
    __syncthreads();
    int c=blockIdx.y*32+lane;
    __nv_bfloat16* dst=c<512?a:b;
    #pragma unroll 1
    for(int q=warp;q<528;q+=8)
        dst[((int64_t)plane*17424+blockIdx.x*528+q)*512+c%512]=transpose[lane][q];
}

std::vector<torch::Tensor> padding(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat &&
                x.sizes()==at::IntArrayRef({8,24576,1024}) &&
                x.strides()==at::IntArrayRef({25165824,1,24576}),
                "fused preparation requires CUDA FP32 [8,24576,1024], channel-strided backbone tokens");
    auto a=torch::empty({192,132,132,512},x.options().dtype(torch::kBFloat16));
    auto b=torch::empty_like(a);
    prepare_reuse<<<dim3(33,32,192),256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {a.permute({0,3,1,2}),b.permute({0,3,1,2})};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("padding", &padding);}
