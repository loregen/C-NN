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
#define NORMALIZE 255

Dataset data_read(char *file_path, int n_examples, DataType data_type, LabelType label_type, int inpsize, int outsize, bool normalize){
    //every line is an example and its label
    //label is first, example is second
    //entries are separeted by one char

    Dataset data;
    data.inpsize = inpsize, data.outsize = outsize;
    data.n_examples = n_examples, data.type = data_type;

    FILE *file;
    if((file = fopen(file_path, "r")) == NULL){
        printf("data_read: could not open file \"%s\".\n", file_path);
        exit(0);
    }

    ALLOCA(data.examples, data.n_examples * data.inpsize, double, "data_read: malloc failed.\n");
    ALLOCA(data.labels, data.n_examples * data.outsize, double, "data_read: malloc failed.\n");
    memset(data.labels, 0, data.n_examples * data.outsize * sizeof(double));

    size_t i, j;
    switch(data.label_type = label_type){
        case NORMAL:{
            for(i = 0; i < data.n_examples; i++){
                for(j = 0; j < data.outsize; j++){
                    fscanf(file, "%lf%*c", &data.labels[data.outsize * i + j]);
                }
                for(j = 0; j < data.inpsize; j++){
                    fscanf(file, "%lf%*c", &data.examples[data.inpsize * i + j]);
                }
            }
            break;
        }
        case ONEHOT:{
            for(i = 0; i < data.n_examples; i++){
                fscanf(file, "%zd%*c", &j);
                data.labels[data.outsize * i + j] = 1;
                for(j = 0; j < data.inpsize; j++){
                    fscanf(file, "%lf%*c", &data.examples[data.inpsize * i + j]);
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

    if(normalize) cblas_dscal(data.n_examples * data.inpsize,
                              1 / (double)NORMALIZE,
                              data.examples,
                              1);

    return data;
}

int read_mnist_txt(double *training_data, double *labels, char *file_name){
    FILE *file;
    if((file = fopen(file_name, "r")) == NULL){
        printf("read_mnist_txt: file error.\n");
        exit(0);
    }
    int n_lines = 0;
    char c; int k;
    while((c = fgetc(file)) != EOF) if(c == '\n') n_lines++;
    rewind(file);
    //printf("%d\n", n_lines);
    memset(labels, 0, n_lines * 10 * sizeof(double));
    for(int i = 0; i < n_lines; i++){
        fscanf(file, "%d", &k);
        (labels + 10 * i)[k] = 1;
        for(int j = 0; j < 784; j++) fscanf(file, ",%lf", training_data + 784 * i + j);
    }

    cblas_dscal(60000 * 784,
                1 / (double)255,
                training_data,
                1);

    fclose(file);
    printf("File read ok.\n");
    return n_lines;
}

void data_free(Dataset *data){
    free(data->examples);
    free(data->labels);
    return;
}