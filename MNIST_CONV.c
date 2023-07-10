#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"

#define N_TRAIN_EX 60000
#define N_TEST_EX 10000
#define N_EPOCHS 6
#define BATCH_SIZE 20
#define LEARNING_RATE 0.18
#define DBG false

#define PATH_TO_TRAIN "MNIST_DATA/MNIST_train.txt"
#define PATH_TO_TEST "MNIST_DATA/MNIST_test.txt"

int main(void){

    srand(time(NULL));

    size_t input_shape[3] = {1, 28, 28};
    Net net = net_create(3, input_shape);
    add_layer(&net, CONV2D, 3, 9, 5, 5);
    add_layer(&net, ACTIVATION_RELU, 1, true);
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
