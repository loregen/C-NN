#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "net.h"

#define PBAR_LENGTH (uint32_t)30

Net net_create(uint8_t input_dim, size_t *input_shape){
    Net net;
    net.n_layers = 0;
    net.trainable_params = 0;
    net.compiled = false;
    net.training_mode = false;
    net.layers = NULL, net.train_info = NULL;

    net.input = Tensor_init(input_dim, input_shape, false, false);
    net.output = zeroTensor;

    return net;
}

void net_free(Net *net){

    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->free(net->layers[i]);
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
    uint8_t dim = (N == 0) ? net->input.dim : net->layers[N - 1]->output.dim;
    size_t *shape = (N == 0) ? net->input.shape : net->layers[N - 1]->output.shape;
    u_int32_t inpsize = (N == 0) ? net->input.size : net->layers[N - 1]->output.size;

    //initialize layer
    switch(type){
        case DENSE:{
            int outsize = va_arg(valist, int);

            net->layers[N] = (Layer *)DenseLayer_init(inpsize, outsize);
            break;
        }
        case ACTIVATION_TANH:{
            net->layers[N] = (Layer *)TanhLayer_init(dim, shape, va_arg(valist, int));
            break;
        }
        case ACTIVATION_SOFTMAX:{
            net->layers[N] = (Layer *)SoftmaxLayer_init(dim, shape, va_arg(valist, int));
            break;
        }
        case ACTIVATION_RELU:{
            net->layers[N] = (Layer *)ReluLayer_init(dim, shape, va_arg(valist, int));
            break;
        }
        case CONV2D:{
            int input_rows = shape[1];
            int input_cols = shape[2];
            int depth = shape[0];

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
        net->layers[i]->print(net->layers[i]);
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

    //set shape and strides of first layer
    net->layers[0]->input = net->input;

    //set input of every layer to output of previous layer
    for(i = 1; i < net->n_layers; i++){
        net->layers[i]->input = net->layers[i - 1]->output;
    }

    //check shapes
    for(i = 0; i < net->n_layers; i++){
        net->layers[i]->check_shapes(net->layers[i]);
    }

    //set net output to output of last layer
    net->output = net->layers[net->n_layers - 1]->output;

    net->compiled = true;
    printf("net compiled successfully\n");

    net_print(net);

    return;
}

void allocate_train_vars(Net *net, Dataset *data, TrainVars *train_vars){

    size_t i;

    //allocate indices for training shuffling
    ALLOCA(train_vars->index, data->n_examples, int, "allocate_train_vars: malloc failed\n");
    for(i = 0; i < data->n_examples; i++) train_vars->index[i] = i;

    //compute number of batches
    train_vars->N_batches = data->n_examples / train_vars->batch_size;

    //allocate gradients inside layers
    for(i = 0; i < net->n_layers; i++){
        net->layers[i]->allocate_grads(net->layers[i]);
        if(i > 0) net->layers[i]->input_grad = net->layers[i - 1]->output_grad;
    }

    net->training_mode = true;

    return;
}

void free_train_vars(Net *net, TrainVars *train_vars){

    free(train_vars->index);

    //free gradients inside layers
    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->free_grads(net->layers[i]);
    }

    net->training_mode = false;

    return;
}

void save_train_info(Net *net, Dataset *data, TrainVars *train_vars){

    ALLOCA(net->train_info, 1, TrainInfo, "save_train_info: malloc failed\n");

    net->train_info->error = train_vars->error;
    net->train_info->learn_rate = train_vars->learn_rate;
    net->train_info->loss_fn = train_vars->loss_fn;
    net->train_info->N_training_ex = data->n_examples;
    net->train_info->batch_siz = train_vars->batch_size;
    net->train_info->N_batches = train_vars->N_batches;

    return;
}

void net_train(Net *net, Dataset *data, uint32_t batch_size, LossType loss_fn, double learn_rate, uint32_t epochs){

    if(net->compiled == false){
        printf("net_train: net must be compiled before training\n");
        exit(0);
    }

    if(data->inpsize != net->input.size || data->outsize != net->output.size){
        printf("net_train: data and net sizes do not match\n");
        exit(0);
    }

    //initialize training variables
    TrainVars train;
    train.loss_fn = loss_fn, train.learn_rate = learn_rate, train.batch_size = batch_size;
    allocate_train_vars(net, data, &train);

    if(data->n_examples % batch_size){
        printf("net_train: batch size must be a multiple of the number of training examples\n");
        exit(0);
    }

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
        Progress progress = {0, i + 1, epochs, &train};
        pthread_t progress_thread;
        pthread_create(&progress_thread, NULL, progress_updater, &progress);

        //set error to 0
        train.error = 0;

        //shuffle training examples
        shuffle((int *)train.index, data->n_examples);

        //batch loop
        for(j = 0; j < train.N_batches; j++){

            //training example loop
            for(k = 0; k < train.batch_size; k++){

                //set input of first layer to current training example
                net->input.data = data->examples + train.index[j * train.batch_size + k] * net->input.size;
                net->layers[0]->input.data = net->input.data;

                //forward loop
                for(l = 0; l < net->n_layers; l++){
                    net->layers[l]->forward(net->layers[l]);
                }
                l--;

                //compute error and gradient of cost
                switch(train.loss_fn){
                    case MSE:{
                        train.error += mse(net->layers[l]->output.data,
                                           data->labels + train.index[j * train.batch_size + k] * net->output.size,
                                           net->output.size);
                        mse_prime(net->layers[l]->output_grad.data,
                                  net->layers[l]->output.data,
                                  data->labels + train.index[j * train.batch_size + k] * net->output.size,
                                  net->output.size);
                        break;
                    }
                    case CROSS_ENTROPY_ONEHOT:{
                        train.error += cross_entropy_onehot(net->layers[l]->output.data,
                                                            data->labels + train.index[j * train.batch_size + k] * net->output.size);
                        cross_entropy_onehot_prime(net->layers[l]->output_grad.data,
                                                   net->layers[l]->output.data,
                                                   data->labels + train.index[j * train.batch_size + k] * net->output.size,
                                                   net->output.size);
                        break;
                    }
                    default: break;
                }

                //backward loop
                for(; l >= 0; l--){
                    net->layers[l]->backward(net->layers[l]);
                }
            }//end of batch
            progress.current_batch++;

            //update loop
            for(l = 0; l < net->n_layers; l++){
                if(net->layers[l]->update != NULL){
                    net->layers[l]->update(net->layers[l], &train);
                }
            }
        }//end of epoch

        //average error
        train.error /= data->n_examples;

        //print progress bar
        pthread_join(progress_thread, NULL);
    }//end of training

    gettimeofday(&end, NULL);
    printf("Total training time: %lf seconds\n\n", (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0);

    save_train_info(net, data, &train);
    free_train_vars(net, &train);

    return;
}

void net_forward(Net *net, double *input, double *output){

    //assume input and output are allocated with net->inpsize and net->outsize and will be freed

    net->input.data = input;
    net->layers[0]->input.data = net->input.data;

    for(size_t i = 0; i < net->n_layers; i++){
        net->layers[i]->forward(net->layers[i]);
    }

    memcpy(output, net->output.data, net->output.size * sizeof(double));

    return;
}

void net_predict(Net *net, Dataset *data){

    if(net->compiled == false){
        printf("net_predict: net must be compiled before prediction\n");
        exit(0);
    }

    double *output;
    ALLOCA(output, net->output.size, double, "net_predict: malloc failed\n");

    double accuracy = 0;

    for(size_t i = 0; i < data->n_examples; i++){

        net_forward(net, data->examples + i * net->input.size, output);

        int label_index = 0;
        while((data->labels + i * net->output.size)[label_index++] != 1.0);
        label_index--;

        int max_output_index = 0;
        double max_output_value = output[0];
        for(size_t j = 1; j < net->output.size; j++){
            if(output[j] > max_output_value){
                max_output_value = output[j];
                max_output_index = j;
            }
        }
        if(max_output_index == label_index) accuracy++;
    }

    printf("Accuracy on test set: %lf\n", accuracy / data->n_examples);

    free(output);
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
