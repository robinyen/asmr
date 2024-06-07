from setuptools import setup, find_packages

setup(
    name="kspace_classifier",
    packages=find_packages(),
    install_requires=[
        "pytorch_lightning >= 1.4.9",
    ],
)
