// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Cirrus-42 (AI agent): share native replicate padding and AMP conversion.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

__global__ void prepare(const float* x, __nv_bfloat16* a, __nv_bfloat16* b) {
    __shared__ __nv_bfloat16 transpose[32][33];
    int lane=threadIdx.x%32, warp=threadIdx.x/32;
    int p=blockIdx.x*32+lane, plane=blockIdx.z;
    int h=max(0,min(127,p/132-2)), w=max(0,min(127,p%132-2));
    #pragma unroll
    for(int j=0;j<4;++j) {
        int c=blockIdx.y*32+warp+j*8;
        float v=x[((int64_t)plane*1024+c)*16384+h*128+w];
        transpose[warp+j*8][lane]=__float2bfloat16_rn(v);
    }
    __syncthreads();
    int c=blockIdx.y*32+lane;
    __nv_bfloat16* dst=c<512?a:b;
    #pragma unroll
    for(int j=0;j<4;++j) {
        int q=blockIdx.x*32+warp+j*8;
        if(q<17424) dst[((int64_t)plane*17424+q)*512+c%512]=transpose[lane][warp+j*8];
    }
}

std::vector<torch::Tensor> padding(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.is_contiguous() &&
                x.dim()==4 && x.size(0)==192 && x.size(1)==1024 && x.size(2)==128 && x.size(3)==128,
                "shared padding requires contiguous CUDA FP32 [192,1024,128,128]");
    auto a=torch::empty({192,132,132,512},x.options().dtype(torch::kBFloat16));
    auto b=torch::empty_like(a);
    prepare<<<dim3(545,32,192),256,0,c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),reinterpret_cast<__nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(b.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {a.permute({0,3,1,2}),b.permute({0,3,1,2})};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("padding", &padding);}
