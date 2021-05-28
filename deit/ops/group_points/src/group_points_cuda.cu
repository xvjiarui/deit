// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu

#include <stdio.h>
#include <stdlib.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <torch/extension.h>

#include "group_points_cuda.cuh"

#define THREADS_PER_BLOCK 512
inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}



void group_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                       int nsample, at::Tensor grad_out_tensor,
                                       at::Tensor idx_tensor, at::Tensor grad_points_tensor) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)

//  dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
//              b);  // blockIdx.x(col), blockIdx.y(row)
//  dim3 threads(THREADS_PER_BLOCK);
//
//  group_points_grad_kernel<<<blocks, threads, 0, stream>>>(
//      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  int num_kernels = npoints * nsample;
  dim3 blocks(GET_BLOCKS(num_kernels), c, b);  // blockIdx.x(col), blockIdx.y(row)


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out_tensor.scalar_type(), "group_points_grad", ([&] {

        scalar_t *grad_points = grad_points_tensor.data_ptr<scalar_t>();
        const int *idx = idx_tensor.data_ptr<int>();
        const scalar_t *grad_out = grad_out_tensor.data_ptr<scalar_t>();

        group_points_grad_kernel<<<blocks,
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            b, c, n, npoints, nsample, grad_out, idx, grad_points);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void group_points_kernel_launcher(int b, int c, int n, int npoints, int nsample,
                                  at::Tensor points_tensor, at::Tensor idx_tensor,
                                  at::Tensor out_tensor) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
//  dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c,
//              b);  // blockIdx.x(col), blockIdx.y(row)
//  dim3 threads(THREADS_PER_BLOCK);
//
//  group_points_kernel<<<blocks, threads, 0, stream>>>(b, c, n, npoints, nsample,
//                                                      points, idx, out);

  int num_kernels = npoints * nsample;
  dim3 blocks(GET_BLOCKS(num_kernels), c, b);  // blockIdx.x(col), blockIdx.y(row)
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points_tensor.scalar_type(), "group_points", ([&] {

        const scalar_t *points = points_tensor.data_ptr<scalar_t>();
        const int *idx = idx_tensor.data_ptr<int>();
        scalar_t *out = out_tensor.data_ptr<scalar_t>();

        group_points_kernel<<<blocks, THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            b, c, n, npoints, nsample, points, idx, out);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
