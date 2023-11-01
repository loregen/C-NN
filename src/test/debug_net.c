#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "net.h"
#include "tensor.h"

#define N_TRAIN_EX 4
#define N_TEST_EX 0
#define N_EPOCHS 30
#define BATCH_SIZE 2
#define LEARNING_RATE 0.1

#define INPSIZE 3
#define OUTSIZE 2


int main(void){

    srand(42);

    size_t input_shape[1] = {INPSIZE};
    Net net = net_create(1, input_shape);
    add_layer(&net, DENSE, 1, 2);
    add_layer(&net, ACTIVATION_TANH, 1, false);
    add_layer(&net, DENSE, 1, OUTSIZE);
    add_layer(&net, ACTIVATION_SOFTMAX, 1, false);
    
    net_compile(&net);

    Dataset train_debug_dataset = {.type = TRAIN,
                                      .n_examples = N_TRAIN_EX,
                                      .example_size = INPSIZE,
                                      .label_size = OUTSIZE,
                                      .label_type = ONEHOT,
                                      .examples = Tensor_init(2, (size_t[]){N_TRAIN_EX, INPSIZE}, true, true),
                                      .labels = Tensor_init(2, (size_t[]){N_TRAIN_EX, OUTSIZE}, true, true)
                                    };

    for(int i = 0; i < N_TRAIN_EX; i++){
        for(int j = 0; j < INPSIZE; j++){
            train_debug_dataset.examples.data[i * INPSIZE + j] = (double)rand() / (double)RAND_MAX;
        }
        train_debug_dataset.labels.data[i * OUTSIZE + rand() % OUTSIZE] = 1;
    }

    net_train(&net, &train_debug_dataset, BATCH_SIZE, CROSS_ENTROPY_ONEHOT, LEARNING_RATE, N_EPOCHS, false);

    data_free(&train_debug_dataset);
    net_free(&net);

    return 0;

}
