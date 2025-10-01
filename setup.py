import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AnyCapture",
    version="0.0.2",
    author="luo3300612, zzaiyan",
    author_email="591486669@qq.com, 1@zzaiyan.com",
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
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "bytecode",
    ],
)