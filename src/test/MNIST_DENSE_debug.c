#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"

#define N_TRAIN_EX 1
#define N_TEST_EX 0
#define N_EPOCHS 1
#define BATCH_SIZE 1
#define LEARNING_RATE 0.05

#define PATH_TO_TRAIN "MNIST_DATA/MNIST_train.txt"
#define PATH_TO_TEST "MNIST_DATA/MNIST_test.txt"

int main(void){

    //srand(time(NULL));

    size_t input_shape[1] = {784};
    Net net = net_create(1, input_shape);
    add_layer(&net, DENSE, 1, 100);
    add_layer(&net, ACTIVATION_TANH, 1, false);
    add_layer(&net, DENSE, 1, 50);
    add_layer(&net, ACTIVATION_TANH, 1, false);
    add_layer(&net, DENSE, 1, 10);
    add_layer(&net, ACTIVATION_SOFTMAX, 1, false);
    
    net_compile(&net);

    Dataset mnist_train_data = data_read(PATH_TO_TRAIN, N_TRAIN_EX, TRAIN, ONEHOT, 1, input_shape, 10);
    Tensor_scale(&mnist_train_data.examples, 1.0 / 255.0);

    net_train(&net, &mnist_train_data, BATCH_SIZE, CROSS_ENTROPY_ONEHOT, LEARNING_RATE, N_EPOCHS, false);

    Dataset mnist_test_data = data_read(PATH_TO_TEST, N_TEST_EX, TEST, ONEHOT, 1, input_shape, 10);
    Tensor_scale(&mnist_test_data.examples, 1.0 / 255.0);

    net_predict(&net, &mnist_test_data);

    printf("Loss on train set: %lf\n", net.train_info->error);

    data_free(&mnist_train_data);
    data_free(&mnist_test_data);
    net_free(&net);

    return 0;

}
