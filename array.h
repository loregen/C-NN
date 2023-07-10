#ifndef ARRAY_H
#define ARRAY_H

void mrand(double* m, int n_rows, int n_cols);
void shuffle(int *pattern, int size);

double mse(double *output, double *label, int siz);
void mse_prime(double *grad, double *output, double *label, int siz);

double cross_entropy_onehot(double *output, double *label);
void cross_entropy_onehot_prime(double *grad, double *output, double *label, int siz);

void full_convolution(double* input_matrix1, double* input_matrix2, int rows1, int cols1, int rows2, int cols2, double* output_matrix);
void valid_cross_correlation(double *matrixA, double *matrixB, double *resultMatrix, int rowsA, int colsA, int rowsB, int colsB);

double array_sum(double *m, int size);

#endif