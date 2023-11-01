#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <string.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
#endif

#include "net.h"

Dataset data_read(char *file_path, uint32_t n_examples, DataType data_type, LabelType label_type, uint8_t example_dim, size_t *example_shape, uint32_t label_size){
    
    FILE *file;
    if((file = fopen(file_path, "r")) == NULL){
        printf("data_read: could not open file \"%s\".\n", file_path);
        exit(0);
    }
    
    //every line is an example and its label
    //label is first, example is second
    //entries are separeted by one char

    Dataset data;
    data.type = data_type;
    data.n_examples = n_examples;
    data.label_size = label_size;
    data.label_type = label_type;

    size_t batched_example_shape[example_dim + 1];
    batched_example_shape[0] = n_examples;
    memcpy(batched_example_shape + 1, example_shape, example_dim * sizeof(size_t));

    data.examples = Tensor_init(example_dim + 1, batched_example_shape, true, true);
    data.example_size = data.examples.strides[0];

    size_t batched_label_shape[2] = {n_examples, label_size};
    data.labels = Tensor_init(2, batched_label_shape, true, true);
    data.label_size = data.labels.strides[0];

    size_t i, j;
    switch(data.label_type = label_type){
        case NORMAL:{
            for(i = 0; i < data.n_examples; i++){
                for(j = 0; j < data.label_size; j++){
                    fscanf(file, "%lf%*c", GetAddress2D(data.labels, i, j));
                }
                for(j = 0; j < data.example_size; j++){
                    fscanf(file, "%lf%*c", GetAddress2D(data.examples, i, j));
                }
            }
            break;
        }
        case ONEHOT:{
            for(i = 0; i < data.n_examples; i++){
                fscanf(file, "%zd%*c", &j);
                GetValue2D(data.labels, i, j) = 1;
                for(j = 0; j < data.example_size; j++){
                    fscanf(file, "%lf%*c", GetAddress2D(data.examples, i, j));
                }
            }
            break;
        }
    }

    if(feof(file)){
        printf("data_read: EOF encountered.\n");
        exit(0);
    }
    fclose(file);

    char *file_name = strrchr(file_path, '/') != NULL ? strrchr(file_path, '/') + 1 : file_path;
    printf("%s: file read ok.\n\n", file_name); 

    return data;
}

void data_free(Dataset *data){
    
    Tensor_free(&data->examples);
    Tensor_free(&data->labels);

    return;
}