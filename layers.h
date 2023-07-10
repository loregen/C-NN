#ifndef LAYERS_H
#define LAYERS_H

#include "macros.h"
#include "tensor.h"
#include "array.h"

typedef struct TrainVars_ TrainVars; // Forward declaration

typedef enum LayerType_{
    DENSE,
    CONV2D,
    MAXPOOL,
    ACTIVATION_TANH,
    ACTIVATION_SIGM,
    ACTIVATION_SOFTMAX,
    ACTIVATION_RELU,
}LayerType;

typedef struct Layer_{
    LayerType type;
    uint32_t index;

    Tensor input, output;
    Tensor input_grad, output_grad;

    uint32_t trainable_params;

    void (*forward)(struct Layer_ *self);
    void (*backward)(struct Layer_ *self);
    void (*update)(struct Layer_ *self, TrainVars *vars);   

    void (*free)(struct Layer_ *self);
    void (*print)(struct Layer_ *self);
    void (*check_shapes)(struct Layer_ *self);
    void (*allocate_grads)(struct Layer_ *self);
    void (*free_grads)(struct Layer_ *self);
}Layer;

typedef struct DenseLayer_{
    Layer base;
    Tensor weights, biases;
    Tensor weights_grad, biases_grad;
}DenseLayer;

typedef Layer TanhLayer;
typedef Layer ReluLayer;
typedef Layer SoftmaxLayer;

typedef Layer FlattenLayer;

typedef struct Conv2DLayer_{
    Layer base;
    uint32_t input_rows, input_cols, depth;
    uint32_t kernel_rows, kernel_cols, n_kernels;

    Tensor kernels, biases;  
    Tensor kernels_grad, biases_grad;
    Tensor im2col;

    uint32_t output_rows, output_cols;

    //strides, repeated for convenience
    uint32_t single_kernel_size; //kernel_rows * kernel_cols
    uint32_t kernel_size; //kernel_rows * kernel_cols * depth
    uint32_t total_kernel_size; //kernel_rows * kernel_cols * depth * n_kernels
    uint32_t single_output_size; //output_rows * output_cols

}Conv2DLayer;

void Layer_print(Layer *base);

DenseLayer *DenseLayer_init(int inpsize, int outsize);
void DenseLayer_allocate_grads(Layer *base);
void DenseLayer_free(Layer *base);
void DenseLayer_free_grads(Layer *base);
void DenseLayer_forward(Layer *base);
void DenseLayer_backward(Layer *base);
void DenseLayer_update(Layer *base, TrainVars *train_vars);
void DenseLayer_check_shapes(Layer *base);

Layer *ActivationLayer_init(uint8_t input_dim, size_t *shape, bool flatten_output);
void ActivationLayer_allocate_grads(Layer *base);
void ActivationLayer_free(Layer *base);
void ActivationLayer_free_grads(Layer *base);;
void ActivationLayer_check_shapes(Layer *base);

TanhLayer *TanhLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output);
void TanhLayer_forward(Layer *base);
void TanhLayer_backward(Layer *base);

SoftmaxLayer *SoftmaxLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output);
void SoftmaxLayer_forward(Layer *base);
void SoftmaxLayer_backward(Layer *base);

ReluLayer *ReluLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output);
void ReluLayer_forward(Layer *base);
void ReluLayer_backward(Layer *base);

Conv2DLayer *Conv2DLayer_init(int input_rows, int input_cols, int depth, int n_kernels, int kernel_rows, int kernel_cols);
void Conv2DLayer_allocate_grads(Layer *base);
void Conv2DLayer_free(Layer *base);
void Conv2DLayer_free_grads(Layer *base);
void Conv2DLayer_forward(Layer *base);
void Conv2DLayer_backward(Layer *base);
void Conv2DLayer_update(Layer *base, TrainVars *train_vars);
void Conv2DLayer_check_shapes(Layer *base);

#endif