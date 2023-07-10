#include <stdlib.h>
#include "tensor.h"
#include "macros.h"

Tensor zeroTensor = {NULL, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0};

Tensor Tensor_init(uint8_t dim, size_t *shape, bool allocate, bool zero){

    Tensor tensor;

    size_t i, size = 1;
    for(i = 0; i < dim; i++){
        tensor.shape[i] = shape[i];
        size *= shape[i];
    }

    tensor.strides[dim-1] = 1;
    for(i = dim - 1; i > 0; i--){
        tensor.strides[i-1] = tensor.strides[i] * tensor.shape[i];
    }

    tensor.dim = dim;
    tensor.size = size;

    //allocate tensor->data 
    if(allocate == false){
        tensor.data = NULL;
    }else if(zero == false){
        ALLOCA(tensor.data, size, double, "Tensor_init: failed to allocate tensor->data\n");
    }else{
        CALLOCA(tensor.data, size, double, "Tensor_init: failed to allocate tensor->data\n");
    }

    return tensor;
}

void Tensor_free(Tensor *tensor){

    free(tensor->data);
    *tensor = zeroTensor;

    return;
}

void Tensor_randomize(Tensor *tensor, double min, double max){
    size_t i;
    for(i = 0; i < tensor->size; i++){
        tensor->data[i] = (double)rand()/(double)(RAND_MAX/(max - min)) + min;
    }
    return;
}

void print_shape(uint8_t dim, size_t *shape){
    printf("(");
    size_t i;
    for(i = 0; i < dim; i++){
        printf("%zu", shape[i]);
        if(i != dim - 1) printf(", ");
    }
    printf(")");
    return;
}

bool Has_shape(Tensor *tensor1, uint8_t dim, size_t *shape){
    if(tensor1->dim != dim) return false;
    size_t i;
    for(i = 0; i < dim; i++){
        if(tensor1->shape[i] != shape[i]) return false;
    }
    return true;
}