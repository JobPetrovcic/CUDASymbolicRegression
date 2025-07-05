import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import shutil

# Get the path to the PyTorch library files (still useful for debugging/info, but not needed for extra_link_args anymore)
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib') # No longer strictly necessary for link args

# ccache support
extra_compile_args = {'cxx': ['-fopenmp'], 'nvcc': ['-Xcompiler', '-fopenmp', '--extended-lambda']}
# if shutil.which('ccache'):
#     print("ccache found, will be used for compilation.")
#     os.environ['CC'] = 'ccache gcc'
#     os.environ['CXX'] = 'ccache g++'
#     extra_compile_args['nvcc'].extend(['-ccbin', 'ccache'])


## Custom build extension to clean before build
#class CleanBuildExt(BuildExtension):
#    def run(self):
#        if os.path.exists(self.build_lib):
#            print(f"cleaning build directory: {self.build_lib}")
#            shutil.rmtree(self.build_lib)
#        if os.path.exists(self.build_temp):
#            print(f"cleaning temp build directory: {self.build_temp}")
#            shutil.rmtree(self.build_temp)
#            
#        
#        # Call the original build_ext command
#        BuildExtension.run(self)


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
            extra_compile_args=extra_compile_args,
            extra_link_args=[f'-Wl,-rpath,{torch_lib_dir}']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)