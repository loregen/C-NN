#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include "pgm.h"
#include "net.h"


void writePGMImage(const char *filename, PGMData *data) {
    FILE *pgmFile = fopen(filename, "w");
    if (pgmFile == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Write the PGM image header to the file
    fprintf(pgmFile, "P2\n");
    fprintf(pgmFile, "%d %d\n", data->col, data->row);
    fprintf(pgmFile, "%d\n", data->max_gray);

    // Write the pixel values from the vector into the file
    for (int i = 0; i < data->row; ++i) {
        for (int j = 0; j < data->col; ++j) {
            fprintf(pgmFile, "%d ", (int)data->vector[i * data->col + j]);
        }
        fprintf(pgmFile, "\n");
    }

    fclose(pgmFile);
}

void writeKernelsToPGM(Net *net, int layer){

    if(net->compiled == false){
        printf("writeKernelsToPGM: net not compiled.\n");
        return;
    }

    Conv2DLayer *conv = (Conv2DLayer *)net->layers[layer];
    int n_kernels = conv->n_kernels;

    PGMData kernels[n_kernels];
    for(int i = 0; i < n_kernels; i++){
        kernels[i].row = conv->kernel_rows;
        kernels[i].col = conv->kernel_cols;
        kernels[i].max_gray = 255;

        ALLOCA(kernels[i].vector, kernels[i].row * kernels[i].col, double, "writeKernelsToPGM: malloc failed.\n");

        memcpy(kernels[i].vector, GetAddress4D(conv->kernels, i, 0, 0, 0), kernels[i].row * kernels[i].col * sizeof(double));

        // rescale to [0, 255] calculating min and max
        double min  = DBL_MAX;
        double max = DBL_MIN;

        for(int j = 0; j < kernels[i].row * kernels[i].col; j++){
            if(kernels[i].vector[j] < min) min = kernels[i].vector[j];
            if(kernels[i].vector[j] > max) max = kernels[i].vector[j];
        }

        for(int j = 0; j < kernels[i].row * kernels[i].col; j++){
            kernels[i].vector[j] = (kernels[i].vector[j] - min) / (max - min) * 255;
        }

        char filename[20];
        sprintf(filename, "kernel_%d.pgm", i);
        writePGMImage(filename, &kernels[i]);

        free(kernels[i].vector);
    }
    return;
}