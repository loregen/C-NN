#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include "layers.h"
#include "net.h"

#define WEIGHTS_MIN -0.1
#define WEIGHTS_MAX 0.1

void Layer_print(Layer *base, bool print_output, bool print_output_grad){

    printf("Layer %d - ", base->index);
    switch(base->type){
        case DENSE:{
            printf("DENSE: ");
            break;
        }
        case CONV2D:{
            printf("CONV2D: ");
            break;
        }
        case MAXPOOL:{
            printf("MAXPOOL: ");
            break;
        }
        case ACTIVATION_TANH:{
            printf("TANH: ");
            break;
        }
        case ACTIVATION_SIGM:{
            printf("SIGM: ");
            break;
        }
        case ACTIVATION_SOFTMAX:{
            printf("SOFTMAX: ");
            break;
        }
        case ACTIVATION_RELU:{
            printf("RELU: ");
            break;
        }
        default:{
            printf("Layer_print: unknown layer type.\n");
            exit(0);
        }
    }
    print_shape(base->input_dim, base->input_shape);
    printf(" --> ");
    print_shape(base->output_dim, base->output_shape);
    printf("   - trainable params: %d\n", base->trainable_params);

    //debug prints
    if(print_output){
        if(base->type == DENSE){
            DenseLayer *dense = (DenseLayer *)base;
            printf("Weights:\n");
            matrix_print(dense->weights.data, dense->weights.shape[0], dense->weights.strides[0]);
            printf("Biases:\n");
            matrix_print(dense->biases.data, dense->biases.shape[0], dense->biases.strides[0]);
        }
        printf("Output:\n");
        matrix_print(base->output.data, base->output.shape[0], base->output.strides[0]);
    }

    if(print_output_grad){
        if(base->type == DENSE){
            DenseLayer *dense = (DenseLayer *)base;
            printf("Weights grad:\n");
            matrix_print(dense->weights_grad.data, dense->weights_grad.shape[0], dense->weights_grad.strides[0]);
            printf("Biases grad:\n");
            matrix_print(dense->biases_grad.data, dense->biases_grad.shape[0], dense->biases_grad.strides[0]);
        }
        printf("Output grad:\n");
        matrix_print(base->output_grad.data, base->output_grad.shape[0], base->output_grad.strides[0]);
    }

    return;
}

void Layer_forward_init(Layer *base, uint32_t batch_size){
    
    size_t batched_output_shape[base->output_dim + 1];
    batched_output_shape[0] = batch_size;
    memcpy(batched_output_shape + 1, base->output_shape, base->output_dim * sizeof(size_t));

    base->output = Tensor_init(base->output_dim + 1, batched_output_shape, true, true);
    return;

}

void Layer_forward_exit(Layer *base){
    Tensor_free(&base->output);
    return;
}

DenseLayer *DenseLayer_init(uint32_t inpsize, uint32_t outsize){
    DenseLayer *dense;
    ALLOCA(dense, 1, DenseLayer, "DenseLayer_init: malloc failed\n");
    Layer *base = (Layer *)dense;

    base->type = DENSE;
    base->index = 0;

    base->forward = &DenseLayer_forward;
    base->backward = &DenseLayer_backward;
    base->update = &DenseLayer_update;

    base->free = &DenseLayer_free;
    base->print = &Layer_print;
    base->check_shapes = &DenseLayer_check_shapes;

    base->forward_init = &Layer_forward_init;
    base->forward_exit = &Layer_forward_exit;
    base->backward_init = &DenseLayer_backward_init;
    base->backward_exit = &DenseLayer_backward_exit;

    base->input_dim = 1;
    base->input_shape[0] = inpsize;
    base->input.dim = 2;
    base->input.shape[1] = inpsize;

    base->output_dim = 1;
    base->output_shape[0] = outsize;

    size_t weights_shape[] = {outsize, inpsize};
    size_t biases_shape[] = {outsize};

    dense->weights = Tensor_init(2, weights_shape, true, false);
    dense->biases = Tensor_init(1, biases_shape, true, false);

    Tensor_randomize(&dense->weights, WEIGHTS_MIN, WEIGHTS_MAX);
    Tensor_randomize(&dense->biases, WEIGHTS_MIN, WEIGHTS_MAX);

    //initialize tensors to NULL
    base->input = zeroTensor;
    base->output = zeroTensor;

    dense->weights_grad = zeroTensor;
    dense->biases_grad = zeroTensor;

    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = dense->weights.size + dense->biases.size;

    return dense;
}
void DenseLayer_backward_init(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);
    
    dense->weights_grad = Tensor_init(dense->weights.dim, dense->weights.shape, true, true);
    dense->biases_grad = Tensor_init(dense->biases.dim, dense->biases.shape, true, true);

    return;
}
void DenseLayer_free(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    Tensor_free(&dense->weights);
    Tensor_free(&dense->biases);

    return;
}
void DenseLayer_backward_exit(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    Tensor_free(&base->output_grad);
    Tensor_free(&dense->weights_grad);
    Tensor_free(&dense->biases_grad);

    return;
}
void DenseLayer_forward(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    //copy biases to output
    for(size_t i = 0; i < base->output.shape[0]; i++){
        memcpy(GetAddress2D(base->output, i, 0),
               dense->biases.data,
               dense->biases.size * sizeof(double));
    }

    //compute output
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                base->output.shape[0],
                base->output.shape[1],
                base->input.shape[1],
                1,
                base->input.data,
                base->input.shape[1],
                dense->weights.data,
                dense->weights.shape[1],
                1,
                base->output.data,
                base->output.shape[1]);
    
    return;
}
void DenseLayer_backward(Layer *base){

    DenseLayer *dense = (DenseLayer *)base;

    //compute weights gradient
    cblas_dgemm(CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                dense->weights_grad.shape[0],
                dense->weights_grad.shape[1],
                base->output_grad.shape[0],
                1,
                base->output_grad.data,
                base->output_grad.shape[1],
                base->input.data,
                base->input.shape[1],
                1,
                dense->weights_grad.data,
                dense->weights_grad.shape[1]);

    //compute biases gradient, TODO: optimize
    for(size_t batch = 0; batch < base->output_grad.shape[0]; batch++){
        cblas_daxpy(dense->biases_grad.size,
                    1,
                    GetAddress2D(base->output_grad, batch, 0),
                    1,
                    dense->biases_grad.data,
                    1);
    }

    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    //compute input gradient
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                base->output_grad.shape[0],
                dense->weights.shape[1],
                base->output_grad.shape[1],
                1,
                base->output_grad.data,
                base->output_grad.shape[1],
                dense->weights.data,
                dense->weights.shape[1],
                0,
                base->input_grad.data,
                base->input_grad.shape[1]);
    
    return;
}
void DenseLayer_update(Layer *base, Net *net){

    DenseLayer *dense = (DenseLayer *)base;

    //update weights
    cblas_daxpy(dense->weights.size,
                -net->train_vars.learn_rate / net->batch_size,
                dense->weights_grad.data,
                1,
                dense->weights.data,
                1);

    //update biases
    cblas_daxpy(dense->biases.size,
                -net->train_vars.learn_rate / net->batch_size,
                dense->biases_grad.data,
                1,
                dense->biases.data,
                1);
    
    //zero out gradients

    memset(dense->weights_grad.data, 0, dense->weights_grad.size * sizeof(double));
    memset(dense->biases_grad.data, 0, dense->biases_grad.size * sizeof(double));

    return;
}
void DenseLayer_check_shapes(Layer *base){

    DenseLayer *dense = (DenseLayer *)base;
    
    size_t expected_input_shape[] = {dense->weights.shape[1]};

    if(compare_shapes(base->input.dim - 1, base->input.shape + 1, 1, expected_input_shape) == false){
        printf("DenseLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input shape ", base->index);
        print_shape(1, expected_input_shape);
        printf(", got ");
        print_shape(base->input.dim - 1, base->input.shape + 1);
        printf(".\n");
        exit(0);
    }

    return;
}

Layer *ActivationLayer_init(uint8_t input_dim, size_t *input_shape){
    Layer *base;
    ALLOCA(base, 1, Layer, "ActivationLayer_init: malloc failed\n");

    base->update = NULL;
    base->forward = NULL;
    base->backward = NULL;
    base->index = 0;

    base->free = NULL;
    base->print = &Layer_print;
    base->check_shapes = &ActivationLayer_check_shapes;

    base->forward_init = &Layer_forward_init;
    base->forward_exit = &Layer_forward_exit;
    base->backward_init = &ActivationLayer_backward_init;
    base->backward_exit = &ActivationLayer_backward_exit;

    base->input_dim = input_dim;
    memcpy(base->input_shape, input_shape, input_dim * sizeof(size_t));
    base->output_dim = input_dim;
    memcpy(base->output_shape, input_shape, input_dim * sizeof(size_t));
    
    //initialize tensors to NULL
    base->input = zeroTensor;
    base->output = zeroTensor;

    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = 0;

    return base;
}
void ActivationLayer_backward_init(Layer *base){
    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);
    return;
}
void ActivationLayer_backward_exit(Layer *base){
    Tensor_free(&base->output_grad);
    return;
}
void ActivationLayer_print(Layer *base){

    switch(base->type){
        case ACTIVATION_TANH:{
            printf("TANH : ");
            break;
        }
        case ACTIVATION_SIGM:{
            printf("SIGM : ");
            break;
        }
        case ACTIVATION_SOFTMAX:{
            printf("SOFTMAX : ");
            break;
        }
        case ACTIVATION_RELU:{
            printf("RELU : ");
            break;
        }
        default:{
            printf("ActivationLayer_print: unknown activation layer type.\n");
            exit(0);
        }
    }
    
    print_shape(base->output.dim, base->output.shape);
    printf("\n");

    return;
}
void ActivationLayer_check_shapes(Layer *base){

    size_t expected_input_shape[base->output_dim];
    memcpy(expected_input_shape, base->output_shape, base->output_dim * sizeof(size_t));

    if(compare_shapes(base->input.dim - 1, base->input.shape + 1, base->output_dim, expected_input_shape) == false){
        printf("ActivationLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input shape ", base->index);
        print_shape(base->output_dim, base->output_shape);
        printf(", got ");
        print_shape(base->input.dim, base->input.shape);
        printf(".\n");
        exit(0);
    }

    return;
}

TanhLayer *TanhLayer_init(uint8_t input_dim, size_t *input_shape){
    
    TanhLayer *base = (TanhLayer *)ActivationLayer_init(input_dim, input_shape);

    base->type = ACTIVATION_TANH;

    base->forward = &TanhLayer_forward;
    base->backward = &TanhLayer_backward;

    return base;
}
void TanhLayer_forward(Layer *base){
    
    for(size_t i = 0; i < base->input.size; i++){
        base->output.data[i] = tanh(base->input.data[i]);
    }

    return;
}
void TanhLayer_backward(Layer *base){

    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    for(size_t i = 0; i < base->input.size; i++){
        base->input_grad.data[i] = base->output_grad.data[i] * (1 - SQUARE(base->output.data[i]));
    }

    return;
}

SoftmaxLayer *SoftmaxLayer_init(uint8_t input_dim, size_t *input_shape){

    SoftmaxLayer *base = (SoftmaxLayer *)ActivationLayer_init(input_dim, input_shape);

    base->type = ACTIVATION_SOFTMAX;

    base->forward = &SoftmaxLayer_forward;
    base->backward = &SoftmaxLayer_backward;

    return base;
}
void SoftmaxLayer_forward(Layer *base){
    double max, sum;

    for (size_t batch = 0; batch < base->input.shape[0]; batch++) {
        max = GetValue2D(base->input, batch, 0);
        sum = 0;

        for(size_t i = 1; i < base->input.shape[1]; i++){
            double value = GetValue2D(base->input, batch, i);
            if(value > max){
                max = value;
            }
        }

        for(size_t i = 0; i < base->input.shape[1]; i++){
            GetValue2D(base->output, batch, i) = exp(GetValue2D(base->input, batch, i) - max);
            sum += GetValue2D(base->output, batch, i);
        }

        sum = 1 / sum;

        for(size_t i = 0; i < base->input.shape[1]; i++){
            GetValue2D(base->output, batch, i) *= sum;
        }
    }

    return;
}
void SoftmaxLayer_backward(Layer *base){
    if(base->index == 0) return; //don't propagate gradient if layer is input layer

    for(size_t batch = 0; batch < base->input_grad.shape[0]; batch++) {
        // Compute the term that is shared by all elements in the updated gradient
        double shared_term = 0.0;
        for(size_t i = 0; i < base->input_grad.shape[1]; i++){
            shared_term += GetValue2D(base->output, batch, i) * GetValue2D(base->output_grad, batch, i);
        }

        // Update the gradient
        for(size_t i = 0; i < base->input_grad.shape[1]; i++){
            GetValue2D(base->input_grad, batch, i) = GetValue2D(base->output, batch, i) * 
                (GetValue2D(base->output_grad, batch, i) - shared_term);
        }
    }
    
    return;
}

ReluLayer *ReluLayer_init(uint8_t input_dim, size_t *input_shape){

    ReluLayer *base = (ReluLayer *)ActivationLayer_init(input_dim, input_shape);

    base->type = ACTIVATION_RELU;

    base->forward = &ReluLayer_forward;
    base->backward = &ReluLayer_backward;

    return base;
}
void ReluLayer_forward(Layer *base){

    for(size_t i = 0; i < base->input.size; i++){
        base->output.data[i] = base->input.data[i] > 0 ? base->input.data[i] : 0;
    }

    return;
}
void ReluLayer_backward(Layer *base){

    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    for(size_t i = 0; i < base->input.size; i++){
        base->input_grad.data[i] = base->output.data[i] > 0 ? base->output_grad.data[i] : 0;
    }

    return;
}

Conv2DLayer *Conv2DLayer_init(int input_rows, int input_cols, int depth, int n_kernels, int kernel_rows, int kernel_cols){
    Conv2DLayer *conv;
    ALLOCA(conv, 1, Conv2DLayer, "Conv2DLayer_init: malloc failed\n");
    Layer *base = (Layer *)conv;

    base->type = CONV2D;
    base->index = 0;

    base->forward = &Conv2DLayer_forward;
    base->backward = &Conv2DLayer_backward;
    base->update = &Conv2DLayer_update;

    base->free = &Conv2DLayer_free;
    base->print = &Layer_print;
    base->check_shapes = &Conv2DLayer_check_shapes;

    base->forward_init = &Layer_forward_init;
    base->forward_exit = &Layer_forward_exit;
    base->backward_init = &Conv2DLayer_backward_init;
    base->backward_exit = &Conv2DLayer_backward_exit;

    conv->input_rows = input_rows;
    conv->input_cols = input_cols;
    conv->depth = depth;

    conv->kernel_rows = kernel_rows;
    conv->kernel_cols = kernel_cols;
    conv->n_kernels = n_kernels;

    conv->single_kernel_size = kernel_rows * kernel_cols;
    conv->kernel_size = conv->single_kernel_size * depth;

    conv->output_rows = input_rows - kernel_rows + 1;
    conv->output_cols = input_cols - kernel_cols + 1;

    conv->single_output_size = conv->output_rows * conv->output_cols;

    base->input_dim = 3;
    base->input_shape[0] = depth;
    base->input_shape[1] = input_rows;
    base->input_shape[2] = input_cols;

    base->output_dim = 3;
    base->output_shape[0] = n_kernels;
    base->output_shape[1] = conv->output_rows;
    base->output_shape[2] = conv->output_cols;

    size_t kernels_shape[] = {n_kernels, depth, kernel_rows, kernel_cols};
    conv->kernels = Tensor_init(4, kernels_shape, true, false);

    size_t biases_shape[] = {n_kernels};
    conv->biases = Tensor_init(1, biases_shape, true, false);

    Tensor_randomize(&conv->kernels, WEIGHTS_MIN, WEIGHTS_MAX);
    Tensor_randomize(&conv->biases, WEIGHTS_MIN, WEIGHTS_MAX);

    //initialize gradients to NULL
    conv->kernels_grad = zeroTensor;
    conv->biases_grad = zeroTensor;

    conv->im2col_forward = zeroTensor;
    conv->im2col_backward = zeroTensor;

    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = conv->kernels.size + conv->biases.size;

    return conv;
}
void Conv2DLayer_backward_init(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);

    conv->kernels_grad = Tensor_init(conv->kernels.dim, conv->kernels.shape, true, true);
    conv->biases_grad = Tensor_init(conv->biases.dim, conv->biases.shape, true, true);

    return;
}  
void Conv2DLayer_free(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    Tensor_free(&conv->kernels);
    Tensor_free(&conv->biases);

    return;
}
void Conv2DLayer_backward_exit(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    Tensor_free(&base->output_grad);
    Tensor_free(&conv->kernels_grad);
    Tensor_free(&conv->biases_grad);

    return;
}
void Conv2DLayer_forward(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    size_t k; //kernel index
    size_t d; //depth index
    size_t batch;

    //copy biases to output
    /*for(batch = 0; batch < base->output.shape[0]; batch++){ 
        for(k = 0; k < conv->n_kernels; k++){
            catlas_dset(base->output.strides[1],
                        conv->biases.data[k],
                        GetAddress4D(base->output, batch, k, 0, 0),
                        1);
        }
    }*/
    
    //convolve
    for(batch = 0; batch < base->output.shape[0]; batch++){
        for(k = 0; k < conv->n_kernels; k++){
            for(d = 0; d < conv->depth; d++){
                valid_cross_correlation(GetAddress4D(base->input, batch, d, 0, 0),
                                        GetAddress4D(conv->kernels, k, d, 0, 0),
                                        GetAddress4D(base->output, batch, k, 0, 0),
                                        conv->input_rows,
                                        conv->input_cols,
                                        conv->kernel_rows,
                                        conv->kernel_cols);
            }
        }
    }
    return;
}
void Conv2DLayer_backward(Layer *base){

    Conv2DLayer *conv = (Conv2DLayer *)base;

    size_t k; //kernel index
    size_t d; //depth index

    //compute kernels gradient
    for(k = 0; k < conv->n_kernels; k++){
        for(d = 0; d < conv->depth; d++){
            valid_cross_correlation(GetAddress3D(base->input, d, 0, 0),
                                    GetAddress3D(base->output_grad, k, 0, 0),
                                    GetAddress4D(conv->kernels_grad, k, d, 0, 0),
                                    conv->input_rows,
                                    conv->input_cols,
                                    conv->output_rows,
                                    conv->output_cols);
        }
    }

    //compute biases gradient
    for(k = 0; k < conv->n_kernels; k++){
        conv->biases_grad.data[k] = array_sum(GetAddress3D(base->output_grad, k, 0, 0), conv->single_output_size);
    }

    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    //zero out gradient
    memset(base->input_grad.data, 0, base->input_grad.size * sizeof(double));

    //compute input gradient
    for(d = 0; d < conv->depth; d++){
        for(k = 0; k < conv->n_kernels; k++){
            full_convolution(GetAddress3D(base->output_grad, k, 0, 0),
                             GetAddress4D(conv->kernels, k, d, 0, 0),
                             conv->output_rows,
                             conv->output_cols,
                             conv->kernel_rows,
                             conv->kernel_cols,
                             GetAddress3D(base->input_grad, d, 0, 0));
        }
    }

    return;
}
void Conv2DLayer_update(Layer *base, Net *net){

    Conv2DLayer *conv = (Conv2DLayer *)base;

    //update kernels
    cblas_daxpy(conv->kernels.size,
                -net->train_vars.learn_rate / net->batch_size,
                conv->kernels_grad.data,
                1,
                conv->kernels.data,
                1);

    //update biases
    cblas_daxpy(conv->n_kernels,
                -net->train_vars.learn_rate / net->batch_size,
                conv->biases_grad.data,
                1,
                conv->biases.data,
                1);
    
    //zero out gradients
    memset(conv->kernels_grad.data, 0, conv->kernels_grad.size * sizeof(double));
    memset(conv->biases_grad.data, 0, conv->biases_grad.size * sizeof(double));

    return;
}
void Conv2DLayer_check_shapes(Layer *base){

    Conv2DLayer *conv = (Conv2DLayer *)base;

    size_t expected_input_shape[] = {conv->depth, conv->input_rows, conv->input_cols};

    if(compare_shapes(base->input.dim - 1, base->input.shape + 1, 3, expected_input_shape) == false){
        printf("Conv2DLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input shape ", base->index);
        print_shape(3, expected_input_shape);
        printf(", got ");
        print_shape(base->input.dim - 1, base->input.shape + 1);
        printf(".\n");
        exit(0);
    }

    return;
}
