#include <math.h>
#include "softmax.h"
#include "tensor.h"

void softmax(tensor *t) {
    int n = t->rows * t->cols;

    // Find max to stabilize
    float max_val = t->data[0];
    for (int i = 1; i < n; i++) {
        if (t->data[i] > max_val) {
            max_val = t->data[i];
        }
    }

    // Exp sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        t->data[i] = expf(t->data[i] - max_val);
        sum += t->data[i];
    }

    // Normalize
    for (int i = 0; i < n; i++) {
        t->data[i] /= sum;
    }
}
