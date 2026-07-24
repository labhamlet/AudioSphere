from setuptools import setup, find_packages
import os

setup(
    name="shroom",
    version="0.1.0",
    description="Modal Acoustics Spherical Harmonics ROOM (MASHROOM), "
                "a Python library for efficient room acoustics modeling, "
                "Ambisonics processing, and rigid sphere microphone array simulations.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Yhonatan Gayer",
    packages=["shroom", "devutils"],
    package_dir={
        "shroom": "src/shroom",  # Points to the actual folder in src
        "devutils": "devutils"  # Points to the folder in root
    },
    package_data={
        "shroom": [
            "data/*.npz",
            "data/*.json",
            "data/*.txt",
            "data/*.md",
            "data/*.wav",
            "data/*.sofa",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy>=1.10.0,<1.15.0",
        "pyroomacoustics",
        "matplotlib",
        "sofar",
        "pyyaml",
    ],
    extras_require={
        "dev": ["pytest", "black", "sounddevice"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
