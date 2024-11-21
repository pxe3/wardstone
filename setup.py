from setuptools import setup, find_packages

setup(
    name="wardstone",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.0.0",
        "torch>=1.7.0"
    ],
    description="An open-source toolkit for language model alignment evaluations.",
    author="Arya Prakash",
    author_email="aryaprak@uw.edu",
    url="https://github.com/pxe3/wardstone",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
