from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_constraints",
    ext_modules=[
        CUDAExtension(
            name="cuda_constraints",
            sources=["cuda_constraints_ext_baseline.cpp", "cuda_constraints_baseline.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)