from setuptools import setup, Extension
import pybind11
import sys
import platform

# Common compile arguments
common_compile_args = []

if sys.platform == 'win32':
    # Windows-specific compile arguments
    extra_compile_args = [
        '/O2',  # Optimization level
        '/std:c++14',  # C++ standard (using C++14 for better compatibility)
        '/EHsc',  # Exception handling model
        '/MP',  # Multi-process compilation
        '/DWIN32_LEAN_AND_MEAN',  # Exclude rarely-used Windows headers
    ]
    extra_link_args = []
else:
    # Linux/Unix compile arguments
    extra_compile_args = [
        '-O3',  # Optimization level
        '-std=c++14',  # C++ standard
        '-fPIC',  # Position independent code
    ]
    extra_link_args = ['-fPIC']

ext_modules = [
    Extension(
        "mesh_processor",
        ["mesh_processor.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True)
        ],
        language='c++',
        extra_compile_args=common_compile_args + extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="mesh_processor",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    python_requires='>=3.6',
)