#ifndef LAYERS_H
#define LAYERS_H

#include "macros.h"
#include "tensor.h"
#include "array.h"

typedef struct Net_ Net; //forward declaration

typedef enum LayerType_{
    DENSE,
    CONV2D,
    MAXPOOL,
    ACTIVATION_TANH,
    ACTIVATION_SIGM,
    ACTIVATION_SOFTMAX,
    ACTIVATION_RELU,
    FLATTEN
}LayerType;

typedef struct Layer_{
    LayerType type;
    uint32_t index;

    //do not include batch size as first dimension, used for shape checking during net compilation
    uint8_t input_dim, output_dim; 
    size_t input_shape[3], output_shape[3];

    Tensor input, output;
    Tensor input_grad, output_grad;

    uint32_t trainable_params;

    void (*forward)(struct Layer_ *self);
    void (*backward)(struct Layer_ *self);
    void (*update)(struct Layer_ *self, Net *net);   

    void (*free)(struct Layer_ *self);
    void (*print)(struct Layer_ *self, bool print_output, bool print_output_grad);
    void (*check_shapes)(struct Layer_ *self);
    
    void (*forward_init)(struct Layer_ *self, uint32_t batch_size);
    void (*backward_init)(struct Layer_ *self);
    void (*forward_exit)(struct Layer_ *self);
    void (*backward_exit)(struct Layer_ *self);
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
    Tensor im2col_forward, im2col_backward;

    uint32_t output_rows, output_cols;

    //strides, repeated for convenience
    uint32_t single_kernel_size; //kernel_rows * kernel_cols
    uint32_t kernel_size; //kernel_rows * kernel_cols * depth
    uint32_t total_kernel_size; //kernel_rows * kernel_cols * depth * n_kernels
    uint32_t single_output_size; //output_rows * output_cols

}Conv2DLayer;

typedef struct DenseParams_{
    uint32_t outsize;
}DenseParams;

typedef struct Conv2DParams_{
    uint32_t n_kernels, kernel_rows, kernel_cols;
}Conv2DParams;

typedef union LayerParams_{
    DenseParams dense;
    Conv2DParams conv2d;
}LayerParams;

void Layer_print(Layer *base, bool print_output, bool print_output_grad);

DenseLayer *DenseLayer_init(uint32_t inpsize, uint32_t outsize);
void DenseLayer_backward_init(Layer *base);
void DenseLayer_free(Layer *base);
void DenseLayer_backward_exit(Layer *base);
void DenseLayer_forward(Layer *base);
void DenseLayer_backward(Layer *base);
void DenseLayer_update(Layer *base, Net *net);
void DenseLayer_check_shapes(Layer *base);

Layer *ActivationLayer_init(uint8_t input_dim, size_t *shape);
void ActivationLayer_backward_init(Layer *base);
void ActivationLayer_free(Layer *base);
void ActivationLayer_backward_exit(Layer *base);
void ActivationLayer_check_shapes(Layer *base);

TanhLayer *TanhLayer_init(uint8_t input_dim, size_t *input_shape);
void TanhLayer_forward(Layer *base);
void TanhLayer_backward(Layer *base);

SoftmaxLayer *SoftmaxLayer_init(uint8_t input_dim, size_t *input_shape);
void SoftmaxLayer_forward(Layer *base);
void SoftmaxLayer_backward(Layer *base);

ReluLayer *ReluLayer_init(uint8_t input_dim, size_t *input_shape);
void ReluLayer_forward(Layer *base);
void ReluLayer_backward(Layer *base);

Conv2DLayer *Conv2DLayer_init(int input_rows, int input_cols, int depth, int n_kernels, int kernel_rows, int kernel_cols);
void Conv2DLayer_backward_init(Layer *base);
void Conv2DLayer_free(Layer *base);
void Conv2DLayer_backward_exit(Layer *base);
void Conv2DLayer_forward(Layer *base);
void Conv2DLayer_backward(Layer *base);
void Conv2DLayer_update(Layer *base, Net *net);
void Conv2DLayer_check_shapes(Layer *base);

#endif