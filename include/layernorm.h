#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

typedef struct {
    tensor *gamma;  // scale
    tensor *beta;   // shift
    int size;
} layernorm;

layernorm *layernorm_new(int size);
void layernorm_free(layernorm *ln);
void layernorm_forward(const layernorm *ln, const tensor *input, tensor *out);

#endif
