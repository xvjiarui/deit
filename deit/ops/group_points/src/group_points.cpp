// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points.cpp

#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

int group_points_wrapper(int b, int c, int n, int npoints, int nsample,
                         at::Tensor points_tensor, at::Tensor idx_tensor,
                         at::Tensor out_tensor);

int group_points_grad_wrapper(int b, int c, int n, int npoints, int nsample,
                              at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                              at::Tensor grad_points_tensor);

void group_points_kernel_launcher(int b, int c, int n, int npoints, int nsample,
                                  at::Tensor points_tensor, at::Tensor idx_tensor,
                                  at::Tensor out_tensor);

void group_points_grad_kernel_launcher(int b, int c, int n, int npoints,
                                       int nsample, at::Tensor grad_out_tensor,
                                       at::Tensor idx_tensor, at::Tensor grad_points_tensor);

int group_points_grad_wrapper(int b, int c, int n, int npoints, int nsample,
                              at::Tensor grad_out_tensor, at::Tensor idx_tensor,
                              at::Tensor grad_points_tensor) {

  group_points_grad_kernel_launcher(b, c, n, npoints, nsample, grad_out_tensor, idx_tensor,
                                    grad_points_tensor);
  return 1;
}

int group_points_wrapper(int b, int c, int n, int npoints, int nsample,
                         at::Tensor points_tensor, at::Tensor idx_tensor,
                         at::Tensor out_tensor) {

  group_points_kernel_launcher(b, c, n, npoints, nsample, points_tensor, idx_tensor, out_tensor);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_points_wrapper, "group_points_wrapper");
  m.def("backward", &group_points_grad_wrapper, "group_points_grad_wrapper");
}
