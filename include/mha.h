#ifndef MHA_H
#define MHA_H

#include "attention.h"
#include "linear.h"
#include "tensor.h"

typedef struct {
    int embed_dim;
    int head_count;
    int head_dim;

    attention_head **heads;
    linear_layer *out_proj;
} multi_head_attention;

multi_head_attention *mha_new(int embed_dim, int head_count);
void mha_free(multi_head_attention *mha);
void mha_forward(const multi_head_attention *mha, const tensor *x, tensor *out);

#endif
