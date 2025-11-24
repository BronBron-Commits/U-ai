#include <stdlib.h>
#include <math.h>
#include "layernorm.h"
#include "tensor.h"

#define LN_EPSILON 1e-5f

layernorm *layernorm_new(int size) {
    layernorm *ln = malloc(sizeof(layernorm));
    ln->size = size;

    ln->gamma = tensor_new(size, 1);
    ln->beta  = tensor_new(size, 1);

    // gamma = 1, beta = 0
    for (int i = 0; i < size; i++) {
        ln->gamma->data[i] = 1.0f;
        ln->beta->data[i]  = 0.0f;
    }

    return ln;
}

void layernorm_free(layernorm *ln) {
    if (!ln) return;
    tensor_free(ln->gamma);
    tensor_free(ln->beta);
    free(ln);
}

void layernorm_forward(const layernorm *ln, const tensor *input, tensor *out) {
    int n = ln->size;

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < n; i++) {
        mean += input->data[i];
    }
    mean /= n;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = input->data[i] - mean;
        var += diff * diff;
    }
    var /= n;

    float inv_std = 1.0f / sqrtf(var + LN_EPSILON);

    // Normalize
    for (int i = 0; i < n; i++) {
        float x_hat = (input->data[i] - mean) * inv_std;
        out->data[i] = x_hat * ln->gamma->data[i] + ln->beta->data[i];
    }
}
