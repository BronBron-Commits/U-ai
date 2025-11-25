#include <stdlib.h>
#include "embedding.h"
#include "init.h"
#include "tensor.h"

embedding_layer *embedding_new(int vocab_size, int embed_dim) {
    embedding_layer *e = malloc(sizeof(embedding_layer));
    e->vocab_size = vocab_size;
    e->embed_dim = embed_dim;

    e->weights = tensor_new(vocab_size, embed_dim);

    // Random initialization
    int total = vocab_size * embed_dim;
    for (int i = 0; i < total; i++) {
        e->weights->data[i] = rand_uniform(-0.1f, 0.1f);
    }

    return e;
}

void embedding_free(embedding_layer *e) {
    if (!e) return;
    tensor_free(e->weights);
    free(e);
}

void embedding_forward(const embedding_layer *e, unsigned char token, tensor *out) {
    // out is shape embed_dim x 1
    for (int i = 0; i < e->embed_dim; i++) {
        out->data[i] = e->weights->data[token * e->embed_dim + i];
    }
}
