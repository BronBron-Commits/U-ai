#include <math.h>
#include "activation.h"

void relu(tensor *t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
    }
}

// GELU approximation (used by GPT-style models)
static float gelu_scalar(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

void gelu(tensor *t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->data[i] = gelu_scalar(t->data[i]);
    }
}
