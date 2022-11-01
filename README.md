# Homework 0
Public repository and stub/testing code for Homework 0 of 10-714.

## Dev Environment
- Windows 11
- Miniconda3 (with Python 3.10.6)
- Visual Studio 2022 (with Windows 10 SDK)
- CMake

## Building Steps

Create virtual environment via miniconda
```shell
conda create -n dlsys python=3.10
```

Install required packages (IPyKernel, NumPy, PyTest, Pybind11, etc.)
```
conda install ipykernel numpy pytest pybind11
```

## Hints

Question2: Loading MNIST data
- `gzip.open(filename, mode)`: open a gziped data file
- `struct.unpack(format, buffer)`: load data from buffer with given format

Question4: SGD for softmax regression
- define a `softmax()` function to calculate normalize exponential
- no need to shuffle data set

Question6: Softmax regression in C++
- Try 1: I am using MinGW and Makefile in repo doesn't work for me, said undefined reference to 'XXX'.
- Try 2: Changed to CMake. My laptop can really do the compilation, but my python has problem loading that compiled module, said 'A dynamic link library (DLL) initialization routine failed'.
- Try 3: Changed to MSVC, suffering from configuring environment variables (Path, INCLUDE & LIB). Be sure CMake can find `cl.exe` (in MSVC), `rc.exe` (in Win10 SDK) and pybind11 (in `site-packages`).
- **Experience may be much better with WSL or Linux Containers.**