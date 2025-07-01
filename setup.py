import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the path to the PyTorch library files
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='symbolic_cuda',
    ext_modules=[
        CUDAExtension(
            name='symbolic_cuda', 
            sources=[
                'symbolic_evaluation.cpp',
                'kernels.cu',
            ],
            # This is the key part!
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)