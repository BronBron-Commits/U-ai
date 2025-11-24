#ifndef MLP_H
#define MLP_H

#include "linear.h"

typedef struct {
    linear_layer *layer1;
    linear_layer *layer2;
} mlp;

mlp *mlp_new(int input_size, int hidden_size);
void mlp_free(mlp *m);
void mlp_forward(const mlp *m, const tensor *input, tensor *out);

#endif
