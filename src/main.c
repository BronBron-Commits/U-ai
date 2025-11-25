#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "mha.h"
#include "init.h"

int main() {
    srand(time(NULL));

    int embed_dim = 4;
    int head_count = 2;

    multi_head_attention *mha = mha_new(embed_dim, head_count);

    // Randomize the out_proj and each attention head
    int total = embed_dim * embed_dim;
    for (int h = 0; h < head_count; h++) {
        for (int i = 0; i < total; i++) {
            mha->heads[h]->Wq->weight->data[i] = rand_uniform(-0.5f, 0.5f);
            mha->heads[h]->Wk->weight->data[i] = rand_uniform(-0.5f, 0.5f);
            mha->heads[h]->Wv->weight->data[i] = rand_uniform(-0.5f, 0.5f);
        }
    }

    // Randomize final projection
    int out_total = (embed_dim * head_count) * embed_dim;
    for (int i = 0; i < out_total; i++) {
        mha->out_proj->weight->data[i] = rand_uniform(-0.3f, 0.3f);
    }

    tensor *x = tensor_new(embed_dim, 1);
    x->data[0] = 1.0f;
    x->data[1] = -0.5f;
    x->data[2] = 2.0f;
    x->data[3] = -1.0f;

    tensor *out = tensor_new(embed_dim, 1);

    mha_forward(mha, x, out);

    printf("MHA output:\n");
    for (int i = 0; i < embed_dim; i++) {
        printf("  y%d = %f\n", i, out->data[i]);
    }

    tensor_free(x);
    tensor_free(out);
    mha_free(mha);

    return 0;
}
