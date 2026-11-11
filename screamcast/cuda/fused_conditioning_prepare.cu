// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Cirrus-42 (AI agent): share native replicate padding and AMP conversion.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

// Fuse align_corners=False interpolation, replicate padding and BF16 layout.
// Native generated interpolation uses horizontal lerps then a vertical lerp.
__global__ void prepare(const float* x, __nv_bfloat16* a, __nv_bfloat16* b) {
    __shared__ __nv_bfloat16 transpose[32][129];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int plane=blockIdx.z;
    #pragma unroll
    for(int t=0;t<4;++t) {
    int p=blockIdx.x*128+t*32+lane;
    int h=max(0,min(127,p/132-2)), w=max(0,min(127,p%132-2));
    float fy=fmaxf((h+0.5f)*0.25f-0.5f,0.0f);
    float fx=fmaxf((w+0.5f)*0.25f-0.5f,0.0f);
    int y0=(int)fy, x0=(int)fx, y1=min(y0+1,31), x1=min(x0+1,31);
    float wy=fy-y0, wx=fx-x0;
    #pragma unroll
    for(int j=0;j<4;++j) {
        int c=blockIdx.y*32+warp+j*8;
        const float* src=x+(int64_t)(plane/24)*25165824+c*24576+(plane%24)*1024;
        float v00=src[y0*32+x0], v01=src[y0*32+x1];
        float v10=src[y1*32+x0], v11=src[y1*32+x1];
        float top=fmaf(v01-v00,wx,v00), bottom=fmaf(v11-v10,wx,v10);
        transpose[warp+j*8][t*32+lane]=__float2bfloat16_rn(fmaf(bottom-top,wy,top));
    }
    }
    __syncthreads();
    int c=blockIdx.y*32+lane;
    __nv_bfloat16* dst=c<512?a:b;
    #pragma unroll
    for(int j=0;j<16;++j) {
        int q=blockIdx.x*128+warp+j*8;
        if(q<17424) dst[((int64_t)plane*17424+q)*512+c%512]=transpose[lane][warp+j*8];
    }
}

std::vector<torch::Tensor> padding(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat &&
                x.sizes()==at::IntArrayRef({8,24576,1024}) &&
                x.strides()==at::IntArrayRef({25165824,1,24576}),
                "fused preparation requires CUDA FP32 [8,24576,1024], channel-strided backbone tokens");
    auto a=torch::empty({192,132,132,512},x.options().dtype(torch::kBFloat16));
    auto b=torch::empty_like(a);
    prepare<<<dim3(137,32,192),256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {a.permute({0,3,1,2}),b.permute({0,3,1,2})};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("padding", &padding);}
