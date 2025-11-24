#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "linear.h"

typedef struct {
    linear_layer *Wq;
    linear_layer *Wk;
    linear_layer *Wv;
    int embed_dim;
} attention_head;

attention_head *attention_new(int embed_dim);
void attention_free(attention_head *att);
void attention_forward(const attention_head *att, const tensor *x, tensor *out);

#endif
