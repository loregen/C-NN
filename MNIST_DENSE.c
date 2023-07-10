#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"

#define N_TRAIN_EX 60000
#define N_TEST_EX 10000
#define N_EPOCHS 20
#define BATCH_SIZE 20
#define LEARNING_RATE 0.05
#define DBG false

#define PATH_TO_TRAIN "MNIST_DATA/MNIST_train.txt"
#define PATH_TO_TEST "MNIST_DATA/MNIST_test.txt"

int main(void){

    srand(time(NULL));

    size_t input_shape[1] = {784};
    Net net = net_create(1, input_shape);
    add_layer(&net, DENSE, 1, 100);
    add_layer(&net, ACTIVATION_RELU, 1, false);
    add_layer(&net, DENSE, 1, 50);
    add_layer(&net, ACTIVATION_RELU, 1, false);
    add_layer(&net, DENSE, 1, 10);
    add_layer(&net, ACTIVATION_SOFTMAX, 1, false);
    
    net_compile(&net);

    Dataset mnist_train_data = data_read(PATH_TO_TRAIN, N_TRAIN_EX, TRAIN, ONEHOT, 784, 10, true);

    net_train(&net, &mnist_train_data, BATCH_SIZE, CROSS_ENTROPY_ONEHOT, LEARNING_RATE, N_EPOCHS);

    Dataset mnist_test_data = data_read(PATH_TO_TEST, N_TEST_EX, TEST, ONEHOT, 784, 10, true);

    net_predict(&net, &mnist_test_data);

    printf("Loss on train set: %lf\n", net.train_info->error);

    data_free(&mnist_train_data);
    data_free(&mnist_test_data);
    net_free(&net);

    return 0;

}
