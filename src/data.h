#ifndef DATA_H
#define DATA_H

#include <stdbool.h>
#include "tensor.h"
#include "macros.h"

typedef enum DataType_{
    TRAIN,
    TEST
}DataType;

typedef enum LabelType_{
    NORMAL,
    ONEHOT
}LabelType;

typedef struct Dataset_{
    DataType type;
    uint32_t n_examples;
    uint32_t example_size, label_size;
    LabelType label_type;
    Tensor examples;
    Tensor labels;
}Dataset;

Dataset data_read(char *file_path, uint32_t n_examples, DataType data_type, LabelType label_type, uint8_t example_dim, size_t *example_shape, uint32_t output_size);
void data_free(Dataset *data);

#endif