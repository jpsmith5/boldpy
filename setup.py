#!/usr/bin/env python3
"""
BoldPy: Organ-Agnostic BOLD MRI Analysis Framework
===================================================

Multi-Layer Concentric Object (MLCO) analysis for BOLD MRI data.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="boldpy",
    version="2.2.1",
    description="Organ-Agnostic BOLD MRI Analysis Framework using Multi-Layer Concentric Objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@institution.edu",
    url="https://github.com/yourusername/boldpy",
    license="MIT",
    
    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.20.0",
        ],
    },
    
    # Entry points for command-line scripts (optional)
    entry_points={
        "console_scripts": [
            "boldpy-analyze=boldpy_analyze:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    
    # Keywords
    keywords="MRI BOLD kidney brain analysis imaging medical",
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)
