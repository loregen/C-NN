#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

//macros assume compatible tensor shapes

#define GetAddress2D(tensor, i, j) ((tensor.data) + (i) * (tensor.strides[0]) + (j))
#define GetAddress3D(tensor, i, j, k) ((tensor.data) + (i) * (tensor.strides[0]) + (j) * (tensor.strides[1]) + (k))
#define GetAddress4D(tensor, i, j, k, l) ((tensor.data) + (i) * (tensor.strides[0]) + (j) * (tensor.strides[1]) + (k) * (tensor.strides[2]) + (l))

#define GetValue2D(tensor, i, j) *(GetAddress2D(tensor, i, j))
#define GetValue3D(tensor, i, j, k) *(GetAddress3D(tensor, i, j, k))
#define GetValue4D(tensor, i, j, k, l) *(GetAddress4D(tensor, i, j, k, l))

#define SHAPE(...) (size_t[]){__VA_ARGS__}

typedef struct Tensor_{
    double *data;
    uint8_t dim;
    size_t shape[4];
    size_t strides[4];
    size_t size;
}Tensor;

extern Tensor zeroTensor;

Tensor Tensor_init(uint8_t dim, size_t *shape, bool allocate, bool zero);
void Tensor_free(Tensor *tensor);
void Tensor_randomize(Tensor *tensor, double min, double max);
void Tensor_scale(Tensor *tensor, double scalar);
void print_shape(uint8_t dim, size_t *shape);
bool Tensor_has_shape(Tensor *tensor1, uint8_t dim, size_t *shape);
bool compare_shapes(uint8_t dim1, size_t *shape1, uint8_t dim2, size_t *shape2);
void Tensor_print(Tensor *tensor, char *message, bool print_data);

#endif