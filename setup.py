#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="ffd_generator",
    version="1.0.0",
    description="FFD (Free-Form Deformation) control box generator for mesh files",
    author="DAFoam Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "meshio": ["meshio>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "ffd_generator=ffd_generator:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
)