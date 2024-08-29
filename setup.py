from os import path, listdir

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("src", name)),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["--expt-extended-lambda"],
        },
        include_dirs=["include/"],
    )

here = path.abspath(path.dirname(__file__))

setuptools.setup(
    # Meta-data
    name="DVT",
    author="Jiayuan Du",
    author_email="dujiayuan@tongji.edu.cn",
    description="DVT Model Code",
    version="1.0.0",
    url="https://github.com/MrPicklesGG/DVT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    python_requires=">=3, <4",

    # Package description
    packages=[
        "dvt",
        "dvt.algos",
        "dvt.config",
        "dvt.data",
        "dvt.models",
        "dvt.modules",
        "dvt.modules.heads",
        "dvt.utils",
        "dvt.utils.bbx",
        "dvt.utils.nms",
        "dvt.utils.parallel",
        "dvt.utils.roi_sampling",
    ],
    ext_modules=[
        make_extension("nms", "dvt.utils"),
        make_extension("bbx", "dvt.utils"),
        make_extension("roi_sampling", "dvt.utils")
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
