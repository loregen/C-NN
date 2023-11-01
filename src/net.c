#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include "net.h"

#define PBAR_LENGTH (uint32_t)30
#define DBG false

TrainVars zeroTrainVars = {0, 0, 0, NULL, 0, 0, 0};

Net net_create(uint8_t input_dim, size_t *input_shape){
    Net net;
    net.n_layers = 0;
    net.trainable_params = 0;
    net.compiled = false;
    net.mode = NONE;
    net.layers = NULL, net.train_info = NULL;
    net.batch_size = 0;
    net.train_vars = zeroTrainVars;

    net.input_dim = input_dim;
    memcpy(net.input_shape, input_shape, input_dim * sizeof(size_t));

    net.input = zeroTensor;
    net.output = zeroTensor;
    net.output_grad = zeroTensor;
    net.labels = zeroTensor;

    return net;
}

void net_free(Net *net){

    for(size_t i = 0; i < net->n_layers; i++){
        if(net->layers[i]->free != NULL) net->layers[i]->free(net->layers[i]);
        free(net->layers[i]);
    }
    free(net->layers);

    if(net->train_info != NULL){
        free(net->train_info);
    }

    return;
}

void add_layer(Net *net, LayerType type, int argc, ...){

    va_list valist;
    va_start(valist, argc);

    int N = net->n_layers++;

    //allocate memory for new layer
    Layer **tmp;
    REALLOCA(net->layers, N + 1, Layer *, tmp, "add_layer: realloc failed\n");

    //compute input dimension and shape
    uint8_t input_dim = (N == 0) ? net->input_dim : net->layers[N - 1]->output_dim;
    size_t *input_shape = (N == 0) ? net->input_shape : net->layers[N - 1]->output_shape;

    //initialize layer
    switch(type){
        case DENSE:{
            uint32_t input_size = input_shape[0];
            uint32_t output_size = va_arg(valist, int);

            net->layers[N] = (Layer *)DenseLayer_init(input_size, output_size);
            break;
        }
        case ACTIVATION_TANH:{
            net->layers[N] = (Layer *)TanhLayer_init(input_dim, input_shape);
            break;
        }
        case ACTIVATION_SOFTMAX:{
            net->layers[N] = (Layer *)SoftmaxLayer_init(input_dim, input_shape);
            break;
        }
        case ACTIVATION_RELU:{
            net->layers[N] = (Layer *)ReluLayer_init(input_dim, input_shape);
            break;
        }
        case CONV2D:{
            int input_rows = input_shape[1];
            int input_cols = input_shape[2];
            int depth = input_shape[0];

            int kernel_rows = va_arg(valist, int);
            int kernel_cols = va_arg(valist, int);
            int n_kernels = va_arg(valist, int);

            net->layers[N] = (Layer *)Conv2DLayer_init(input_rows, input_cols, depth, kernel_rows, kernel_cols, n_kernels);

            break;
        }
        default:{
            printf("add_layer: invalid layer type\n");
            exit(0);
        }
    }

    //increment trainable parameters
    net->trainable_params += net->layers[N]->trainable_params;

    //set layer index
    net->layers[N]->index = N;

    va_end(valist);
    return;
}

void net_print(Net *net){
    
    if(net->compiled == false){
        printf("net_print: net must be compiled before printing\n");
        exit(0);
    }

    printf("\n");
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->print(net->layers[i], false, false);
    }

    printf("Total trainable parameters: %d\n\n", net->trainable_params);

    return;
}

void net_compile(Net *net){

    if(net->n_layers == 0){
        printf("net_compile: net has no layers\n");
        exit(0);
    }

    if(net->layers[0]->type != DENSE && net->layers[0]->type != CONV2D){
        printf("net_compile: first layer must be dense or convolutional\n");
        exit(0);
    }

    size_t i;

    //set dimension and shape of first layer
    net->layers[0]->input_dim = net->input_dim;
    net->layers[0]->input.dim = net->input_dim + 1;
    memcpy(net->layers[0]->input_shape, net->input_shape, net->input_dim * sizeof(size_t));
    memcpy(net->layers[0]->input.shape + 1, net->input_shape, net->input_dim * sizeof(size_t));

    //set dimensions and shapes of layers
    for(i = 1; i < net->n_layers; i++){
        net->layers[i]->input_dim = net->layers[i - 1]->output_dim;
        net->layers[i]->input.dim = net->layers[i - 1]->output_dim + 1;
        memcpy(net->layers[i]->input_shape, net->layers[i - 1]->output_shape, net->layers[i - 1]->output_dim * sizeof(size_t));
        memcpy(net->layers[i]->input.shape + 1, net->layers[i - 1]->output_shape, net->layers[i - 1]->output_dim * sizeof(size_t));
    }

    //check shapes
    for(i = 0; i < net->n_layers; i++){
        net->layers[i]->check_shapes(net->layers[i]);
    }

    //set net output to output of last layer
    net->output_dim = net->layers[net->n_layers - 1]->output_dim;
    memcpy(net->output_shape, net->layers[net->n_layers - 1]->output_shape, net->output_dim * sizeof(size_t));

    net->compiled = true;
    printf("net compiled successfully\n");

    net_print(net);

    return;
}

void net_train_init(Net *net, Dataset *train_data, uint32_t batch_size, LossType loss_fn, double learn_rate){
    
    if(net->compiled == false){
        printf("net_train_init: net must be compiled before training\n");
        exit(0);
    }

    if(compare_shapes(net->input_dim, net->input_shape, train_data->examples.dim - 1, train_data->examples.shape + 1) == false){
        printf("net_train_init: input shape of net and training data do not match\n");
        exit(0);
    }

    if(compare_shapes(net->output_dim, net->output_shape, train_data->labels.dim - 1, train_data->labels.shape + 1) == false){
        printf("net_train_init: output shape of net and training labels do not match\n");
        exit(0);
    }

    net->batch_size = batch_size;

    TrainVars *vars = &net->train_vars;
    vars->error = 0;
    vars->learn_rate = learn_rate;
    vars->loss_fn = loss_fn;
    vars->example_size = train_data->examples.strides[0];
    vars->label_size = train_data->labels.strides[0];

    if(train_data->n_examples % batch_size != 0){
        printf("net_train_init: batch size must divide number of training examples\n");
        exit(0);
    }
    vars->N_batches = train_data->n_examples / batch_size;

    ALLOCA(vars->index, train_data->n_examples, int, "net_train_init: malloc failed\n");
    for(size_t i = 0; i < train_data->n_examples; i++) vars->index[i] = i;

    //allocate net input
    size_t batched_input_shape[net->input_dim + 1];
    batched_input_shape[0] = batch_size;
    memcpy(batched_input_shape + 1, net->input_shape, net->input_dim * sizeof(size_t));

    net->input = Tensor_init(net->input_dim + 1, batched_input_shape, true, true);

    //allocate memory for labels
    size_t batched_label_shape[net->output_dim + 1];
    batched_label_shape[0] = batch_size;
    memcpy(batched_label_shape + 1, net->output_shape, net->output_dim * sizeof(size_t));

    net->labels = Tensor_init(net->output_dim + 1, batched_label_shape, true, true);

    //allocate output inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward_init(net->layers[i], batch_size);
        if(i == 0) net->layers[i]->input = net->input; //set input of first layer to net input
        else net->layers[i]->input = net->layers[i - 1]->output; //set input of layer to output of previous layer
    }
    net->output = net->layers[net->n_layers - 1]->output; //set net output to output of last layer

    //allocate gradients inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->backward_init(net->layers[i]);
        if(i > 0) net->layers[i]->input_grad = net->layers[i - 1]->output_grad; //set input gradient of layer to output gradient of previous layer
    }

    //set output gradient of net to output gradient of last layer
    net->output_grad = net->layers[net->n_layers - 1]->output_grad;

    net->mode = TRAINING;
    return;
}

void net_train_exit(Net *net){

    //free net train indices
    free(net->train_vars.index);

    //free net input
    Tensor_free(&net->input);

    //free labels and set output to zeroTensor
    Tensor_free(&net->labels);

    net->output = zeroTensor;
    net->output_grad = zeroTensor;

    //free outputs and grads inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward_exit(net->layers[i]);
        net->layers[i]->backward_exit(net->layers[i]);
    }

    net->train_vars = zeroTrainVars;
    net->batch_size = 0;
    net->mode = NONE;
    return;
}

void save_train_info(Net *net, Dataset *data){

    ALLOCA(net->train_info, 1, TrainInfo, "save_train_info: malloc failed\n");

    TrainVars *train_vars = &net->train_vars;

    net->train_info->error = train_vars->error;
    net->train_info->learn_rate = train_vars->learn_rate;
    net->train_info->loss_fn = train_vars->loss_fn;
    net->train_info->N_training_ex = data->n_examples;
    net->train_info->batch_siz = net->batch_size;
    net->train_info->N_batches = train_vars->N_batches;

    return;
}

void net_train(Net *net, Dataset *data, uint32_t batch_size, LossType loss_fn, double learn_rate, uint32_t epochs, bool progress_bar){

    //initialize training variables
    net_train_init(net, data, batch_size, loss_fn, learn_rate);

    size_t i; //epochs
    size_t j; //batches
    size_t k; //training examples
    int l; //layers

    //train timing
    struct timeval start, end;
    gettimeofday(&start, NULL);

    printf("Training...\n");
    //train loop
    for(i = 0; i < epochs; i++){

        //initialize progress bar updater
        Progress progress = {0, i + 1, epochs, &net->train_vars};
        pthread_t progress_thread;
        if(progress_bar == true) pthread_create(&progress_thread, NULL, progress_updater, &progress);

        //set error to 0
        net->train_vars.error = 0;

        //shuffle training examples
        shuffle((int *)net->train_vars.index, data->n_examples);

        //batch loop
        for(j = 0; j < net->train_vars.N_batches; j++, progress.current_batch++){

            //copy current batch to net input and labels 
            for(k = 0; k < net->batch_size; k++){
                memcpy(GetAddress2D(net->input, k, 0), 
                       GetAddress2D(data->examples, net->train_vars.index[j * net->batch_size + k], 0),
                       net->train_vars.example_size * sizeof(double));
                
                memcpy(GetAddress2D(net->labels, k, 0),
                       GetAddress2D(data->labels, net->train_vars.index[j * net->batch_size + k], 0),
                       net->train_vars.label_size * sizeof(double));
            }

            //debug print
            if(DBG == true){
                printf("\nEpoch %zu/%u, Batch %zu/%u\n", i + 1, epochs, j + 1, net->train_vars.N_batches);
                printf("Input: \n");
                matrix_print(net->input.data, net->batch_size, net->train_vars.example_size);
            }

            //forward loop
            for(l = 0; l < net->n_layers; l++){
                net->layers[l]->forward(net->layers[l]);
                if(DBG == true) net->layers[l]->print(net->layers[l], true, false);
            }
            l--;

            //debug print
            if(DBG == true){
                printf("Labels: \n");
                matrix_print(net->labels.data, net->batch_size, net->train_vars.label_size);
            }

            //compute error and gradient of cost
            switch(net->train_vars.loss_fn){
                case CROSS_ENTROPY_ONEHOT:{
                    for(k = 0; k < net->batch_size; k++){
                        net->train_vars.error += cross_entropy_onehot(GetAddress2D(net->output, k, 0),
                                                            GetAddress2D(net->labels, k, 0));
                        cross_entropy_onehot_prime(GetAddress2D(net->output_grad, k, 0),
                                                   GetAddress2D(net->output, k, 0),
                                                   GetAddress2D(net->labels, k, 0),
                                                   net->train_vars.label_size);
                        if(DBG == true) printf("Error: %lf\n", net->train_vars.error);
                    }
                    break;
                }
                default: break;
            }

            //backward loop
            for(; l >= 0; l--){
                net->layers[l]->backward(net->layers[l]);
                if(DBG == true) net->layers[l]->print(net->layers[l], true, true);
            }

            //update loop
            for(l = 0; l < net->n_layers; l++){
                if(net->layers[l]->update != NULL){
                    net->layers[l]->update(net->layers[l], net);
                }
            }
        }//end of epoch

        //average error
        net->train_vars.error /= data->n_examples;
        
        //print progress bar
        if(progress_bar == true) pthread_join(progress_thread, NULL);
    }//end of training

    gettimeofday(&end, NULL);
    printf("Total training time: %lf seconds\n\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

    save_train_info(net, data);
    net_train_exit(net);

    return;
}

void net_inference_init(Net *net, Dataset *data){

    if(net->compiled == false){
        printf("net_inference_init: net must be compiled before inference\n");
        exit(0);
    }

    if(compare_shapes(net->input_dim, net->input_shape, data->examples.dim - 1, data->examples.shape + 1) == false){
        printf("net_inference_init: input shape of net and training data do not match\n");
        exit(0);
    }

    if(compare_shapes(net->output_dim, net->output_shape, data->labels.dim - 1, data->labels.shape + 1) == false){
        printf("net_train_init: output shape of net and training labels do not match\n");
        exit(0);
    }

    net->batch_size = data->n_examples;

    //allocate net input
    size_t batched_input_shape[net->input_dim + 1];
    batched_input_shape[0] = net->batch_size;
    memcpy(batched_input_shape + 1, net->input_shape, net->input_dim * sizeof(size_t));

    net->input = Tensor_init(net->input_dim + 1, batched_input_shape, true, true);

    //allocate memory for labels
    size_t batched_label_shape[net->output_dim + 1];
    batched_label_shape[0] = data->n_examples;
    memcpy(batched_label_shape + 1, net->output_shape, net->output_dim * sizeof(size_t));

    net->labels = Tensor_init(net->output_dim + 1, batched_label_shape, true, true);

    //allocate output inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward_init(net->layers[i], net->batch_size);
        if(i == 0) net->layers[i]->input = net->input;
        else net->layers[i]->input = net->layers[i - 1]->output;
    }
    net->output = net->layers[net->n_layers - 1]->output;

    net->mode = INFERENCE;
    return;
}

void net_inference_exit(Net *net){

    //free net input
    Tensor_free(&net->input);

    //free labels and set output to zeroTensor
    Tensor_free(&net->labels);
    net->output = zeroTensor;

    //free outputs inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward_exit(net->layers[i]);
    }

    net->batch_size = 0;
    net->mode = NONE;
    return;
}

void net_predict(Net *net, Dataset *data){

    //set up net for inference
    net_inference_init(net, data);

    //copy data to net input
    memcpy(net->input.data, data->examples.data, data->examples.size * sizeof(double));

    //copy labels to net labels
    memcpy(net->labels.data, data->labels.data, data->labels.size * sizeof(double));

    //forward loop
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward(net->layers[i]);
    }

    uint32_t label_size = net->output.strides[0];

    //compute accuracy
    uint32_t correct = 0;
    uint32_t incorrect = 0;
    uint32_t predicted;
    uint32_t actual;

    for(size_t examples = 0; examples < data->n_examples; examples++){

        predicted = cblas_idamax(label_size, GetAddress2D(net->output, examples, 0), 1);
        actual = cblas_idamax(label_size, GetAddress2D(net->labels, examples, 0), 1);

        if(predicted == actual) correct++;
        else incorrect++;
    }
    printf("Accuracy: %lf\n", (double)correct / (double)(correct + incorrect));

    net_inference_exit(net);
    return;
}

void *progress_updater(void *arg) {
    Progress *progress = (Progress *)arg;
    TrainVars *train_vars = progress->train_vars;
    size_t i;
    
    printf("Epoch %u/%u\n", progress->current_epoch, progress->total_epochs);

    while (1) {
        printf("\r%4d/%4d [", progress->current_batch, train_vars->N_batches);
        for (i = 0; i < (progress->current_batch * PBAR_LENGTH / train_vars->N_batches); i++) {
            printf("-");
        }
        if (progress->current_batch == train_vars->N_batches) {
            for (; i < PBAR_LENGTH; i++) {
                printf("-");
            }
        } else {
            printf(">");
            for (; i < PBAR_LENGTH - 1; i++) {
                printf(" ");
            }
        }
        printf("]");
        fflush(stdout);
        if (progress->current_batch == train_vars->N_batches) {
            printf(" - loss = %lf\n", train_vars->error);
            break;
        }
        usleep(50000); // Sleep for 50 milliseconds
    }

    printf("\n");
    
    return NULL;
}
