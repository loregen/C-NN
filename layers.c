#include <stdlib.h>
#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include "layers.h"
#include "net.h"

#define WEIGHTS_MIN -0.1
#define WEIGHTS_MAX 0.1

void Layer_print(Layer *base){

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
    print_shape(base->input.dim, base->input.shape);
    printf(" --> ");
    print_shape(base->output.dim, base->output.shape);
    printf("   - trainable params: %d\n", base->trainable_params);

    return;
}

DenseLayer *DenseLayer_init(int inpsize, int outsize){
    DenseLayer *dense;
    ALLOCA(dense, 1, DenseLayer, "DenseLayer_init: malloc failed\n");
    Layer *base = (Layer *)dense;

    base->type = DENSE;

    base->forward = &DenseLayer_forward;
    base->backward = &DenseLayer_backward;
    base->update = &DenseLayer_update;

    base->free = &DenseLayer_free;
    base->print = &Layer_print;
    base->check_shapes = &DenseLayer_check_shapes;
    base->allocate_grads = &DenseLayer_allocate_grads;
    base->free_grads = &DenseLayer_free_grads;

    size_t output_shape[] = {outsize};
    size_t weights_shape[] = {outsize, inpsize};

    base->output = Tensor_init(1, output_shape, true, false);

    dense->weights = Tensor_init(2, weights_shape, true, false);
    dense->biases = Tensor_init(1, output_shape, true, false);

    Tensor_randomize(&dense->weights, WEIGHTS_MIN, WEIGHTS_MAX);
    Tensor_randomize(&dense->biases, WEIGHTS_MIN, WEIGHTS_MAX);

    //initialize gradients to NULL
    dense->weights_grad = zeroTensor;
    dense->biases_grad = zeroTensor;

    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = dense->weights.size + dense->biases.size;

    return dense;
}
void DenseLayer_allocate_grads(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);

    dense->weights_grad = Tensor_init(dense->weights.dim, dense->weights.shape, true, true);
    dense->biases_grad = Tensor_init(dense->biases.dim, dense->biases.shape, true, true);

    return;
}
void DenseLayer_free(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    Tensor_free(&base->output);
    Tensor_free(&dense->weights);
    Tensor_free(&dense->biases);

    free(dense);

    return;
}
void DenseLayer_free_grads(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    Tensor_free(&base->output_grad);
    Tensor_free(&dense->weights_grad);
    Tensor_free(&dense->biases_grad);

    return;
}
void DenseLayer_forward(Layer *base){
    DenseLayer *dense = (DenseLayer *)base;

    memcpy(base->output.data, dense->biases.data, base->output.size * sizeof(double));

    cblas_dgemv(CblasRowMajor,
                CblasNoTrans,
                base->output.size,
                base->input.size,
                1,
                dense->weights.data,
                base->input.size,
                base->input.data,
                1,
                1,
                base->output.data,
                1);
    
    return;
}
void DenseLayer_backward(Layer *base){

    DenseLayer *dense = (DenseLayer *)base;

    //compute weights gradient
    cblas_dger(CblasRowMajor,
               base->output.size,
               base->input.size,
               1,
               base->output_grad.data,
               1,
               base->input.data,
               1,
               dense->weights_grad.data,
               base->input.size);

    //compute biases gradient
    cblas_daxpy(base->output.size,
                1,
                base->output_grad.data,
                1,
                dense->biases_grad.data,
                1);
    
    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    //compute input gradient
    cblas_dgemv(CblasRowMajor,
                CblasTrans,
                base->output.size,
                base->input.size,
                1,
                dense->weights.data,
                base->input.size,
                base->output_grad.data,
                1,
                0,
                base->input_grad.data,
                1);
    
    return;
}
void DenseLayer_update(Layer *base, TrainVars *train_vars){

    DenseLayer *dense = (DenseLayer *)base;

    //update weights
    cblas_daxpy(dense->weights.size,
                -train_vars->learn_rate / train_vars->batch_size,
                dense->weights_grad.data,
                1,
                dense->weights.data,
                1);

    //update biases
    cblas_daxpy(base->output.size,
                -train_vars->learn_rate / train_vars->batch_size,
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

    size_t input_shape[] = {dense->weights.shape[1]};

    if(!Has_shape(&base->input, 1, input_shape)){
        printf("DenseLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input shape ", base->index);
        print_shape(1, input_shape);
        printf(", got ");
        print_shape(base->input.dim, base->input.shape);
        printf(".\n");
        if(base->input.dim > 1){
            printf("Maybe you need to flatten the input before DenseLayer. This can be done in activation layer.\n");
        }
        exit(0);
    }

    return;
}

Layer *ActivationLayer_init(uint8_t input_dim, size_t *shape, bool flatten_output){
    Layer *base;
    ALLOCA(base, 1, Layer, "ActivationLayer_init: malloc failed\n");

    base->update = NULL;

    base->free = &ActivationLayer_free;
    base->print = &Layer_print;
    base->check_shapes = &ActivationLayer_check_shapes;
    base->allocate_grads = &ActivationLayer_allocate_grads;
    base->free_grads = &ActivationLayer_free_grads;

    if(flatten_output){
        size_t output_shape[] = {shape[0] * shape[1] * shape[2]};
        base->output = Tensor_init(1, output_shape, true, false);
    }
    else{
        base->output = Tensor_init(input_dim, shape, true, false);
    }
    
    //initialize gradients to NULL
    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = 0;

    return base;
}
void ActivationLayer_allocate_grads(Layer *base){
    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);
    return;
}
void ActivationLayer_free(Layer *base){
    
    Tensor_free(&base->output);

    free(base);
    return;
}
void ActivationLayer_free_grads(Layer *base){
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

    if(base->input.size != base->output.size){
        printf("ActivationLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input size %zu, got %zu\n", base->index, base->output.size, base->input.size);
        exit(0);
    }
    return;
}

TanhLayer *TanhLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output){
    
    TanhLayer *base = (TanhLayer *)ActivationLayer_init(input_dim, input_shape, flatten_output);

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

SoftmaxLayer *SoftmaxLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output){

    SoftmaxLayer *base = (SoftmaxLayer *)ActivationLayer_init(input_dim, input_shape, flatten_output);

    base->type = ACTIVATION_SOFTMAX;

    base->forward = &SoftmaxLayer_forward;
    base->backward = &SoftmaxLayer_backward;

    return base;
}
void SoftmaxLayer_forward(Layer *base){

    static double aux[2];

    size_t i;
    aux[0] = base->input.data[0];
    aux[1] = 0;

    for(i = 1; i < base->input.size; i++){
        if(base->input.data[i] > aux[0]){
            aux[0] = base->input.data[i];
        }
    }

    for(i = 0; i < base->input.size; i++){
        base->output.data[i] = exp(base->input.data[i] - aux[0]);
        aux[1] += base->output.data[i];
    }
    aux[1] = 1 / aux[1];

    for(i = 0; i < base->input.size; i++){
        base->output.data[i] *= aux[1];
    }

    return;
}
void SoftmaxLayer_backward(Layer *base){

    //don't propagate gradient if layer is input layer
    if(base->index == 0) return;

    size_t i;

    // Compute the term that is shared by all elements in the updated gradient
    double shared_term = 0.0;
    for(i = 0; i < base->input.size; i++){
        shared_term += base->output.data[i] * base->output_grad.data[i];
    }

    // Update the gradient
    for(i = 0; i < base->input.size; i++){
        base->input_grad.data[i] = base->output.data[i] * (base->output_grad.data[i] - shared_term);
    }

    return;
}

ReluLayer * ReluLayer_init(uint8_t input_dim, size_t *input_shape, bool flatten_output){

    ReluLayer *base = (ReluLayer *)ActivationLayer_init(input_dim, input_shape, flatten_output);

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

    base->forward = &Conv2DLayer_forward;
    base->backward = &Conv2DLayer_backward;
    base->update = &Conv2DLayer_update;

    base->free = &Conv2DLayer_free;
    base->print = &Layer_print;
    base->check_shapes = &Conv2DLayer_check_shapes;
    base->allocate_grads = &Conv2DLayer_allocate_grads;
    base->free_grads = &Conv2DLayer_free_grads;

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

    size_t output_shape[] = {n_kernels, conv->output_rows, conv->output_cols};
    base->output = Tensor_init(3, output_shape, true, false);

    size_t kernels_shape[] = {n_kernels, depth, kernel_rows, kernel_cols};
    conv->kernels = Tensor_init(4, kernels_shape, true, false);

    size_t biases_shape[] = {n_kernels};
    conv->biases = Tensor_init(1, biases_shape, true, false);

    /*size_t im2*/

    Tensor_randomize(&conv->kernels, WEIGHTS_MIN, WEIGHTS_MAX);
    Tensor_randomize(&conv->biases, WEIGHTS_MIN, WEIGHTS_MAX);

    //initialize gradients to NULL
    conv->kernels_grad = zeroTensor;
    conv->biases_grad = zeroTensor;

    base->input_grad = zeroTensor;
    base->output_grad = zeroTensor;

    base->trainable_params = conv->kernels.size + conv->biases.size;

    return conv;
}
void Conv2DLayer_allocate_grads(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    base->output_grad = Tensor_init(base->output.dim, base->output.shape, true, true);

    conv->kernels_grad = Tensor_init(conv->kernels.dim, conv->kernels.shape, true, true);
    conv->biases_grad = Tensor_init(conv->biases.dim, conv->biases.shape, true, true);

    return;
}  
void Conv2DLayer_free(Layer *base){
    Conv2DLayer *conv = (Conv2DLayer *)base;

    Tensor_free(&base->output);
    Tensor_free(&conv->kernels);
    Tensor_free(&conv->biases);

    free(conv);

    return;
}
void Conv2DLayer_free_grads(Layer *base){
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

    //copy biases to output
    for(k = 0; k < conv->n_kernels; k++){
        catlas_dset(conv->single_output_size,
                    conv->biases.data[k],
                    GetAddress3D(base->output, k, 0, 0),
                    1);
    }
    
    //convolve
    for(k = 0; k < conv->n_kernels; k++){
        for(d = 0; d < conv->depth; d++){
            valid_cross_correlation(GetAddress3D(base->input, d, 0, 0),
                                    GetAddress4D(conv->kernels, k, d, 0, 0),
                                    GetAddress3D(base->output, k, 0, 0),
                                    conv->input_rows,
                                    conv->input_cols,
                                    conv->kernel_rows,
                                    conv->kernel_cols);
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
void Conv2DLayer_update(Layer *base, TrainVars *train_vars){

    Conv2DLayer *conv = (Conv2DLayer *)base;

    //update kernels
    cblas_daxpy(conv->kernels.size,
                -train_vars->learn_rate / train_vars->batch_size,
                conv->kernels_grad.data,
                1,
                conv->kernels.data,
                1);

    //update biases
    cblas_daxpy(conv->n_kernels,
                -train_vars->learn_rate / train_vars->batch_size,
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

    size_t input_shape[] = {conv->depth, conv->input_rows, conv->input_cols};

    if(!Has_shape(&base->input, 3, input_shape)){
        printf("Conv2DLayer_check_shapes: input shape mismatch.\n");
        printf("In layer %d, expected input shape ", base->index);
        print_shape(3, input_shape);
        printf(", got ");
        print_shape(base->input.dim, base->input.shape);
        printf("\n");
        exit(0);
    }

    return;
}
