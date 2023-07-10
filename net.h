#ifndef NET_H
#define NET_H

#include "layers.h"
#include "data.h"

typedef enum LossType_{
    MSE,
    CROSS_ENTROPY_NORMAL,
    CROSS_ENTROPY_ONEHOT
}LossType;

typedef struct TrainInfo_{
    double error, learn_rate;
    LossType loss_fn;
    uint32_t N_training_ex, batch_siz, N_batches;
}TrainInfo;

typedef struct Net_{
    uint8_t n_layers;
    Layer **layers;

    Tensor input, output;

    uint32_t trainable_params;
    bool compiled;
    bool training_mode; //if true, gradients are allocated inside layers for training

    TrainInfo *train_info;
}Net;

typedef struct TrainVars_{
    double error, learn_rate;
    LossType loss_fn;

    uint32_t *index; //training data indices
    uint32_t batch_size, N_batches;
}TrainVars;

typedef struct Progress_{
    uint32_t current_batch;
    uint32_t current_epoch;
    uint32_t total_epochs;
    TrainVars *train_vars;
}Progress;

Net net_create(uint8_t input_dim, size_t *input_shape);
void net_free(Net *net);
void add_layer(Net *net, LayerType type, int argc, ...);
void net_print(Net *net);
void net_compile(Net *net);
void allocate_train_vars(Net *net, Dataset *data, TrainVars *train_vars);
void free_train_vars(Net *net, TrainVars *train_vars);
void save_train_info(Net *net, Dataset *data, TrainVars *train_vars);
void net_train(Net *net, Dataset *data, uint32_t batch_size, LossType loss_fn, double learn_rate, uint32_t epochs);
void net_forward(Net *net, double *input, double *output);
void net_predict(Net *net, Dataset *data);
void *progress_updater(void *arg);

#endif
