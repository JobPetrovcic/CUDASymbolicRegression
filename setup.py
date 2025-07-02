import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the path to the PyTorch library files
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

setup(
    name='symbolic_torch',
    packages=['symbolic_torch'],
    package_data={'symbolic_torch': ['*.pyi']},
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='symbolic_torch._C', 
            sources=[
                'src/kernels_cuda.cu',
                'src/kernels_cpu.cpp',
                'src/symbolic_evaluation.cpp',
            ],
            include_dirs=['include'],
            extra_compile_args={'cxx': ['-fopenmp'], 'nvcc': ['-Xcompiler', '-fopenmp']},
            # This is the key part!
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)