#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tensor.h"
#include "attention.h"
#include "init.h"

int main() {
    srand(time(NULL));

    attention_head *att = attention_new(4);

    // Random init Wq, Wk, Wv
    int total = att->embed_dim * att->embed_dim;
    for (int i = 0; i < total; i++) {
        att->Wq->weight->data[i] = rand_uniform(-0.5f, 0.5f);
        att->Wk->weight->data[i] = rand_uniform(-0.5f, 0.5f);
        att->Wv->weight->data[i] = rand_uniform(-0.5f, 0.5f);
    }

    tensor *x = tensor_new(4, 1);
    x->data[0] = 1.0f;
    x->data[1] = 0.5f;
    x->data[2] = -1.0f;
    x->data[3] = 2.0f;

    tensor *out = tensor_new(4, 1);

    attention_forward(att, x, out);

    printf("Attention output:\n");
    for (int i = 0; i < 4; i++) {
        printf("  y%d = %f\n", i, out->data[i]);
    }

    tensor_free(x);
    tensor_free(out);
    attention_free(att);

    return 0;
}
