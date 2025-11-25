#include <stdlib.h>
#include "mha.h"
#include "attention.h"
#include "tensor.h"
#include "linear.h"

multi_head_attention *mha_new(int embed_dim, int head_count) {
    multi_head_attention *m = malloc(sizeof(multi_head_attention));
    m->embed_dim = embed_dim;
    m->head_count = head_count;
    m->head_dim = embed_dim; // for now 1-token attention, simple case

    m->heads = malloc(sizeof(attention_head*) * head_count);
    for (int i = 0; i < head_count; i++) {
        m->heads[i] = attention_new(embed_dim);
    }

    m->out_proj = linear_new(embed_dim * head_count, embed_dim);
    return m;
}

void mha_free(multi_head_attention *m) {
    if (!m) return;

    for (int i = 0; i < m->head_count; i++) {
        attention_free(m->heads[i]);
    }
    free(m->heads);

    linear_free(m->out_proj);
    free(m);
}

void mha_forward(const multi_head_attention *m, const tensor *x, tensor *out) {
    int H = m->head_count;
    int D = m->embed_dim;

    // Temporary buffer for concatenated heads
    tensor *concat = tensor_new(D * H, 1);

    // Run each head
    for (int h = 0; h < H; h++) {
        tensor *tmp = tensor_new(D, 1);
        attention_forward(m->heads[h], x, tmp);

        // write to concat
        for (int i = 0; i < D; i++) {
            concat->data[h * D + i] = tmp->data[i];
        }

        tensor_free(tmp);
    }

    // Final projection
    linear_forward(m->out_proj, concat, out);

    tensor_free(concat);
}
