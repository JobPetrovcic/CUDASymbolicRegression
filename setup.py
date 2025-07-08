# setup.py
import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import platform

# NO ccache logic here. It's handled by your environment.

extra_compile_args : dict[str, list[str]] = {
    'cxx': ['-fopenmp'],
    'nvcc': ['-Xcompiler', '-fopenmp', '--extended-lambda']
}
extra_link_args : list[str] = []

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
if platform.system() == 'Darwin':
    extra_link_args.append(f'-Wl,-rpath,{torch_lib_dir}')
else: # Linux
    extra_link_args.append(f'-Wl,-rpath={torch_lib_dir}')

setup(
    name='symbolic_torch',
    packages=['symbolic_torch'],
    package_data={'symbolic_torch': ['*.pyi']},
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='symbolic_torch._C',
            sources=[
                'src/main.cpp',
                'src/evaluation_kernels_cuda.cu',
                'src/evaluation_kernels_cpu.cpp',
                'src/symbolic_evaluation.cpp',
                'src/pcfg.cpp',
                'src/pcfg_cpu.cpp',
                'src/pcfg_cuda.cu',
            ],
            include_dirs=[os.path.abspath('include')],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)