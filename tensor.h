#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

//macros assume compatible tensor shapes

#define GetAddress2D(tensor, i, j) ((tensor.data) + (i) * (tensor.strides[0]) + (j))
#define GetAddress3D(tensor, i, j, k) ((tensor.data) + (i) * (tensor.strides[0]) + (j) * (tensor.strides[1]) + (k))
#define GetAddress4D(tensor, i, j, k, l) ((tensor.data) + (i) * (tensor.strides[0]) + (j) * (tensor.strides[1]) + (k) * (tensor.strides[2]) + (l))

typedef struct Tensor_{
    double *data;
    u_int8_t dim;
    size_t shape[4];
    size_t strides[4];
    size_t size;
}Tensor;

Tensor zeroTensor;

Tensor Tensor_init(uint8_t dim, size_t *shape, bool allocate, bool zero);
void Tensor_free(Tensor *tensor);
void Tensor_randomize(Tensor *tensor, double min, double max);
void print_shape(uint8_t dim, size_t *shape);
bool Has_shape(Tensor *tensor1, uint8_t dim, size_t *shape);

#endif