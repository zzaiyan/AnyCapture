import re
import os
from setuptools import setup, find_packages


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    with open(os.path.join(package, '__init__.py')) as f:
        init_py = f.read()
    return re.search("__version__ = [\'\"]([^\'\"]+)[\'\"]", init_py).group(1)


version = get_version('anycapture')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anycapture",
    version=version,
    author="Zaiyan Zhang",
    author_email="1@zzaiyan.com",
    description="A tool to capture local variables from any function, especially useful for visualizing attention maps in deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzaiyan/AnyCapture",
    project_urls={
        "Original Project": "https://github.com/luo3300612/Visualizer",
        "Bug Tracker": "https://github.com/zzaiyan/AnyCapture/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "bytecode",
    ],
)
