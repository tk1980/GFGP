#include <torch/extension.h>

#include <vector>

// CUDA kernel function declarations

void maxent_pool2d_forward_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode) ;

void maxent_pool2d_backward_cuda(
    torch::Tensor& gradInput,
    torch::Tensor& gradParam,
    const torch::Tensor& gradOutput,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode) ;

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor maxent_pool2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode)
{
  CHECK_INPUT(input) ;
  CHECK_INPUT(param) ;
  torch::Tensor output = at::empty({0}, input.options()) ;

  maxent_pool2d_forward_cuda(output, input, param, kernel_size, stride, padding, ceil_mode) ;

  return output ;
}

std::vector<torch::Tensor> maxent_pool2d_backward(
    const torch::Tensor& gradOutput,
    const torch::Tensor& input,
    const torch::Tensor& param,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode)
{
  CHECK_INPUT(gradOutput) ;
  CHECK_INPUT(input) ;
  CHECK_INPUT(param) ;
  
  torch::Tensor gradInput = at::empty({0}, input.options()) ;
  torch::Tensor gradParam = at::empty({0}, param.options()) ;

  maxent_pool2d_backward_cuda(gradInput, gradParam, gradOutput, input, param, kernel_size, stride, padding, ceil_mode) ;

  return {gradInput, gradParam} ;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &maxent_pool2d_forward, "Maximum-Entropy Pooling forward (CUDA)") ;
  m.def("backward", &maxent_pool2d_backward, "Maximum-Entropy Pooling backward (CUDA)") ;
}
