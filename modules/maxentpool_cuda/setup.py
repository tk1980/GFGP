from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='maxentpool2d_cuda',
    ext_modules=[
        CUDAExtension(
            name = 'maxentpool2d_cuda', 
            sources = [ 'maxentpool2d_cuda.cpp', 'maxentpool2d_cuda_kernel.cu'],
            extra_compile_args={'cxx':[], 'nvcc': ['-arch=sm_60']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
