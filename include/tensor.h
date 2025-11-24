#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int rows;
    int cols;
    float *data;
} tensor;

tensor *tensor_new(int rows, int cols);
void tensor_free(tensor *t);

#endif

void tensor_matmul(
    const tensor *a,
    const tensor *b,
    tensor *out
);

