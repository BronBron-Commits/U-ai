#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"

typedef struct {
    tensor *weight;   // shape: out_features x in_features
    tensor *bias;     // shape: out_features x 1
    int in_features;
    int out_features;
} linear_layer;

linear_layer *linear_new(int in_features, int out_features);
void linear_free(linear_layer *layer);
void linear_forward(const linear_layer *layer, const tensor *input, tensor *out);

#endif
