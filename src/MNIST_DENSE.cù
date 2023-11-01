#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"

#define N_TRAIN_EX 60000
#define N_TEST_EX 10000
#define N_EPOCHS 20
#define BATCH_SIZE 24
#define LEARNING_RATE 0.05

#define PATH_TO_TRAIN "../MNIST_DATA/MNIST_train.txt"
#define PATH_TO_TEST "../MNIST_DATA/MNIST_test.txt"

int main(void){

    srand(time(NULL));

    Net net = net_create(1, SHAPE(784));
    add_layer(&net, DENSE, 1, 40);
    add_layer(&net, ACTIVATION_TANH, 0);
    add_layer(&net, DENSE, 1, 10);
    add_layer(&net, ACTIVATION_SOFTMAX, 0);
    
    net_compile(&net);

    Dataset mnist_train_data = data_read(PATH_TO_TRAIN, N_TRAIN_EX, TRAIN, ONEHOT, 1, SHAPE(784), 10);
    Tensor_scale(&mnist_train_data.examples, 1.0 / 255.0);

    net_train(&net, &mnist_train_data, BATCH_SIZE, CROSS_ENTROPY_ONEHOT, LEARNING_RATE, N_EPOCHS, true);

    Dataset mnist_test_data = data_read(PATH_TO_TEST, N_TEST_EX, TEST, ONEHOT, 1, SHAPE(784), 10);
    Tensor_scale(&mnist_test_data.examples, 1.0 / 255.0);

    net_predict(&net, &mnist_test_data);

    printf("Loss on train set: %lf\n", net.train_info->error);

    data_free(&mnist_train_data);
    data_free(&mnist_test_data);
    net_free(&net);
    
    return 0;

}
