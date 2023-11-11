import glob
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    extension = CppExtension

    extensions_dir = os.path.join("maskrcnn_benchmark", "csrc")
    include_dirs = [extensions_dir]

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    sources = main_file + source_cpu

    define_macros = []
    extra_compile_args = {"cxx": []}

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension

        source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    return [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension.with_options(use_ninja=False)},
    )
