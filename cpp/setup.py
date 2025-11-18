from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

try:
    import pybind11
except Exception as e:
    raise RuntimeError("pybind11 is required to build snake_env_cpp. Install with pip.")

ext_modules = [
    Extension(
        name="snake_env_cpp",
        sources=["snake_env.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="rl-snake",
    version="0.1.0",
    description="RL Snake environment and agents",
    packages=find_packages(include=["rl_snake", "agents", "scripts"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)