from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='symbolic_cuda',
    ext_modules=[
        CUDAExtension('symbolic_cuda', [
            'symbolic_evaluation.cpp',
            'kernels.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })