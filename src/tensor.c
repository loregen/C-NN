#include <stdlib.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include "tensor.h"
#include "macros.h"

Tensor zeroTensor = {NULL, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0};

Tensor Tensor_init(uint8_t dim, size_t *shape, bool allocate, bool zeroes){

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
    }else if(zeroes == false){
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

void Tensor_scale(Tensor *tensor, double scalar){

    cblas_dscal(tensor->size, scalar, tensor->data, 1);
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

bool Tensor_has_shape(Tensor *tensor1, uint8_t dim, size_t *shape){
    if(tensor1->dim != dim) return false;
    size_t i;
    for(i = 0; i < dim; i++){
        if(tensor1->shape[i] != shape[i]) return false;
    }
    return true;
}

bool compare_shapes(uint8_t dim1, size_t *shape1, uint8_t dim2, size_t *shape2){
    if(dim1 != dim2) return false;
    size_t i;
    for(i = 0; i < dim1; i++){
        if(shape1[i] != shape2[i]) return false;
    }
    return true;
}

// Function to print 1D Tensor
void Print1D(Tensor *tensor) {
    for (size_t i = 0; i < tensor->shape[0]; i++)
        printf("%5.3f ", tensor->data[i]);
    printf("\n");
}

// Function to print 2D Tensor
void Print2D(Tensor *tensor) {
    for (size_t i = 0; i < tensor->shape[0]; i++) {
        for (size_t j = 0; j < tensor->shape[1]; j++)
            printf("%5.3f ", tensor->data[i * tensor->shape[1] + j]);
        printf("\n");
    }
}

// Function to print 3D Tensor
void Print3D(Tensor *tensor) {
    for (size_t i = 0; i < tensor->shape[1]; i++) {
        for (size_t j = 0; j < tensor->shape[0]; j++) {
            for (size_t k = 0; k < tensor->shape[2]; k++)
                printf("%5.3f ", tensor->data[j * tensor->shape[1] * tensor->shape[2] + i * tensor->shape[2] + k]);
            
            // Print spaces to separate the matrices
            if (j < tensor->shape[0] - 1)
                printf("   ");  // Three spaces as separation
        }
        printf("\n");
    }
}

// Function to print 4D Tensor
void Print4D(Tensor *tensor) {
    for (size_t p = 0; p < tensor->shape[0]; p++) {
        for (size_t i = 0; i < tensor->shape[2]; i++) {
            for (size_t j = 0; j < tensor->shape[1]; j++) {
                for (size_t k = 0; k < tensor->shape[3]; k++)
                    printf("%5.3f ", tensor->data[p * tensor->shape[1] * tensor->shape[2] * tensor->shape[3] + j * tensor->shape[2] * tensor->shape[3] + i * tensor->shape[3] + k]);
                
                // Print spaces to separate the matrices
                if (j < tensor->shape[1] - 1)
                    printf("   ");  // Three spaces as separation
            }
            printf("\n");
        }
        if (p < tensor->shape[0] - 1)
            printf("\n");  // Adding an extra line to visually separate the 'row' of 3D tensors
    }
}

// Our main Tensor_print function remains unchanged
void Tensor_print(Tensor *tensor, char *message, bool print_data){
    printf("%s: ", message);
    print_shape(tensor->dim, tensor->shape);
    printf("\n");

    if(print_data){
        switch (tensor->dim) {
            case 1:
                Print1D(tensor);
                break;
            case 2:
                Print2D(tensor);
                break;
            case 3:
                Print3D(tensor);
                break;
            case 4:
                Print4D(tensor);
                break;
            default:
                printf("Dimension %d is not supported for printing.\n", tensor->dim);
                break;
        }
    }
    printf("\n");
}

    