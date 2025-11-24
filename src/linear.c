#include <stdlib.h>
#include "linear.h"
#include "tensor.h"

linear_layer *linear_new(int in_features, int out_features) {
    linear_layer *layer = malloc(sizeof(linear_layer));
    if (!layer) return NULL;

    layer->in_features = in_features;
    layer->out_features = out_features;

    layer->weight = tensor_new(out_features, in_features);
    layer->bias   = tensor_new(out_features, 1);

    return layer;
}

void linear_free(linear_layer *layer) {
    if (!layer) return;
    tensor_free(layer->weight);
    tensor_free(layer->bias);
    free(layer);
}

void linear_forward(const linear_layer *layer, const tensor *input, tensor *out) {
    // Expect input shape: (in_features x 1)
    if (!layer || !input || !out) return;
    if (input->rows != layer->in_features || input->cols != 1) return;
    if (out->rows != layer->out_features || out->cols != 1) return;

    // out = W * x
    tensor_matmul(layer->weight, input, out);

    // out += bias
    for (int i = 0; i < layer->out_features; i++) {
        out->data[i] += layer->bias->data[i];
    }
}
