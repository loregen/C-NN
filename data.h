#ifndef DATA_H
#define DATA_H

#include <stdbool.h>
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
    uint32_t inpsize, outsize;
    LabelType label_type;
    double *examples;
    double *labels;
}Dataset;

Dataset data_read(char *file_path, int n_examples, DataType data_type, LabelType label_type, int inpsize, int outsize, bool normalize);
int read_mnist_txt(double *training_data, double *labels, char *file_name);
void data_free(Dataset *data);

#endif