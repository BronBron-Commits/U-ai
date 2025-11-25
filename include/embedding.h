#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
    int vocab_size;
    int embed_dim;
    tensor *weights; // shape: vocab_size x embed_dim
} embedding_layer;

embedding_layer *embedding_new(int vocab_size, int embed_dim);
void embedding_free(embedding_layer *e);

// Lookup embedding vector for a single token ID.
void embedding_forward(const embedding_layer *e, unsigned char token, tensor *out);

#endif
