from setuptools import setup, find_packages

setup(
    name="pt_tnn",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision",
        "networkx",
        "numpy",
        "setuptools",
        "torchviz",
    ],
    python_requires=">=3.7",
)
