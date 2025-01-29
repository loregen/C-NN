# C-NN

C-NN is a neural network library in C, supporting Dense and Convolutional networks. It includes ready-to-use examples for the MNIST dataset.

## Features

- Dense and Convolutional layer support  
- MNIST dataset examples  
- BLAS-accelerated performance  

## Setup

### Requirements

- C compiler (GCC recommended)  
- CMake (version 3.10 or higher)  
- BLAS library (e.g., OpenBLAS, ATLAS, or Accelerate on macOS)  

### MNIST Dataset

Unzip the MNIST dataset in the root directory and ensure it's in `.txt` format.
The compressed dataset is provided in the repository for convenience, but you can also download it from the following link: [NIST](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip).

### Configuration

Set the dataset path in `MNIST_DENSE.c`.

### Compilation

Before compiling, create a `build` directory and run CMake:

```bash
mkdir build
cd build
cmake ..
make
```

For a debug build, use:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make