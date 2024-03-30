# C-NN

C-NN is a neural network library in C, supporting Dense and Convolutional networks. It includes ready-to-use examples for the MNIST dataset.

## Features

- Dense and Convolutional layer support
- MNIST dataset examples
- BLAS-accelerated performance

## Setup

### Requirements

- C compiler (GCC recommended)
- BLAS library. On **Linux**, any implementation will work (e.g. OpenBLAS, ATLAS). On **MacOS**, Accelerate framework is used by default.

### MNIST Dataset

Download from [NIST](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip) and ensure it's in `.txt` format.

### Configuration

Set the dataset path in `MNIST_DENSE.c`.

### Compilation
Use `make` to compile the project:
    
```bash
make
```

For a debug build, use:

```bash
make debug
```
