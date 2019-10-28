import torch
from torch import nn
from torch.autograd import Function

from .build import maxentpool2d_cuda

class maxentpool2d_func(Function):
    @staticmethod
    def forward(ctx, input, weights, kernel_size, stride, padding, ceil_mode):
        output = maxentpool2d_cuda.forward(input, weights, kernel_size, stride, padding, ceil_mode)
        ctx.save_for_backward(input, weights)
        ctx.params = [kernel_size, stride, padding, ceil_mode]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d_input, d_weights = maxentpool2d_cuda.backward( grad_output.contiguous(),  *ctx.saved_variables, *ctx.params)
        return d_input, d_weights, None, None, None, None
