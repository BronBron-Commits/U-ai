#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"

// Simple ReLU
void relu(tensor *t);

// GELU (approximate)
void gelu(tensor *t);

#endif
