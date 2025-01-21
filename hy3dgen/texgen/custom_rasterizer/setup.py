from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# build custom rasterizer
# build with `python setup.py install`
# nvcc is needed

custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
])

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
