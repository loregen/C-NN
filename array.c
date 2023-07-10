#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "array.h"

#define TOLERANCE 1e-2

#define SQUARE(x) ((x)*(x))

void shuffle(int *pattern, int size) {
    int i;

    // Initialize the array with sequential integers
    for (i = 0; i < size; i++) {
        pattern[i] = i;
    }

    // Seed the random number generator
    uint32_t seed = time(NULL);

    // Perform the Fisher-Yates shuffle
    for (i = size - 1; i > 0; i--) {
        int j = rand_r(&seed) % (i + 1);
        if (j != i) {
            int swap = pattern[j];
            pattern[j] = pattern[i];
            pattern[i] = swap;
        }
    }
    return;
}

inline double mse(double *output, double *label, int siz){
    double err = 0;
    for(int i = 0; i < siz; i++) err += SQUARE(output[i] - label[i]);
    return err / (double)siz;
}

inline void mse_prime(double *grad, double *output, double *label, int siz){
    for(int i = 0; i < siz; i++) grad[i] = 2.0 * (output[i] - label[i]) / (double)siz;
    return;
}

inline double cross_entropy_onehot(double *output, double *label){
        int i;
        for(i = 0; label[i] == 0; ++i);
        return -log(output[i]);
}

inline void cross_entropy_onehot_prime(double *grad, double *output, double *label, int siz){
    for(int i = 0; i < siz; i++){
        if(label[i] == 0) grad[i] = 0;
        else grad[i] = - 1 / (output[i] < TOLERANCE ? TOLERANCE : output[i]);
    }
    return;
}

void full_convolution(double* input_matrix1, double* input_matrix2, int rows1, int cols1, int rows2, int cols2, double* output_matrix){
    int i, j, k, l;

    // Iterate over all possible shifts
    for (i = -rows2 + 1; i < rows1; i++) {
        for (j = -cols2 + 1; j < cols1; j++) {
            // Compute the sum of products for this shift
            double sum = 0.0;
            for (k = 0; k < rows2; k++) {
                for (l = 0; l < cols2; l++) {
                    int row1 = i + k;
                    int col1 = j + l;
                    int row2 = rows2 - 1 - k;
                    int col2 = cols2 - 1 - l;
                    if (row1 >= 0 && row1 < rows1 && col1 >= 0 && col1 < cols1 && row2 >= 0 && row2 < rows2 && col2 >= 0 && col2 < cols2) {
                        int index1 = row1 * cols1 + col1;
                        int index2 = row2 * cols2 + col2;
                        sum += input_matrix1[index1] * input_matrix2[index2];
                    }
                }
            }
            // Store the result
            int index = (i + rows2 - 1) * (cols1 + cols2 - 1) + (j + cols2 - 1);
            output_matrix[index] += sum;
        }
    }
}


void valid_cross_correlation(double *matrixA, double *matrixB, double *resultMatrix, int rowsA, int colsA, int rowsB, int colsB){
    int i, j, k, l;
    double sum;

    for(i = 0; i <= rowsA - rowsB; i++) {
        for(j = 0; j <= colsA - colsB; j++) {
            sum = 0;

            for(k = 0; k < rowsB; k++) {
                for(l = 0; l < colsB; l++) {
                    sum += matrixA[(i + k) * colsA + (j + l)] * matrixB[k * colsB + l];
                }
            }

            resultMatrix[i * (colsA - colsB + 1) + j] += sum;
        }
    }
}

double array_sum(double *m, int size){
    double sum = 0;
    for(int i = 0; i < size; i++) sum += m[i];
    return sum;
}

