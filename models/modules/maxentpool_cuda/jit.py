from torch.utils.cpp_extension import load
maxentpool2d_cuda = load(name='maxentpool2d_cuda', sources=['maxentpool2d_cuda.cpp', 'maxentpool2d_cuda_kernel.cu'], verbose=True)
help(maxentpool2d_cuda)
