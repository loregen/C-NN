#ifndef NET_H
#define NET_H

#include "layers.h"
#include "data.h"

typedef enum LossType_{
    MSE,
    CROSS_ENTROPY_NORMAL,
    CROSS_ENTROPY_ONEHOT
}LossType;

typedef enum NetMode_{
    NONE,
    INFERENCE, //outputs inside each layers, net input and labels are allocated
    TRAINING   //gradients and outputs inside each layer, net input and labels are allocated
}NetMode;

typedef struct TrainInfo_{
    double error, learn_rate;
    LossType loss_fn;
    uint32_t N_training_ex, batch_siz, N_batches;
}TrainInfo;

typedef struct TrainVars_{
    double error, learn_rate;
    LossType loss_fn;

    uint32_t *index; //training data indices
    uint32_t N_batches;
    uint32_t example_size, label_size;

}TrainVars;

extern TrainVars zeroTrainVars;

typedef struct Net_{
    uint8_t n_layers;
    Layer **layers;

    //do not include batch size as first dimension, used for shape checking during net compilation
    uint8_t input_dim, output_dim;
    size_t input_shape[3], output_shape[3];
    size_t batch_size;

    Tensor input, output;
    Tensor labels, output_grad;

    uint32_t trainable_params;
    bool compiled;
    NetMode mode;

    TrainVars train_vars;
    TrainInfo *train_info;
}Net;

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
void net_train_init(Net *net, Dataset *train_data, uint32_t batch_size, LossType loss_fn, double learn_rate);
void net_train_exit(Net *net);
void save_train_info(Net *net, Dataset *data);
void net_train(Net *net, Dataset *data, uint32_t batch_size, LossType loss_fn, double learn_rate, uint32_t epochs, bool progress_bar);
void net_inference_init(Net *net, Dataset *data);
void net_inference_exit(Net *net);
void net_predict(Net *net, Dataset *data);
void *progress_updater(void *arg);

#endif
