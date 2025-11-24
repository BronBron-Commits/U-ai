#include <stdlib.h>
#include "tensor.h"

tensor *tensor_new(int rows, int cols) {
    tensor *t = malloc(sizeof(tensor));
    t->rows = rows;
    t->cols = cols;
    t->data = calloc(rows * cols, sizeof(float));
    return t;
}

void tensor_free(tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

void tensor_matmul(
    const tensor *a,
    const tensor *b,
    tensor *out
) {
    if (a->cols != b->rows) return;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;

            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i * a->cols + k] *
                       b->data[k * b->cols + j];
            }

            out->data[i * out->cols + j] = sum;
        }
    }
}

