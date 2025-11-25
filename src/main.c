#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tokenizer.h"
#include "embedding.h"
#include "tensor.h"

int main() {
    srand(time(NULL));

    const char *text = "Hi!";
    unsigned char tokens[64];
    size_t count = tokenize(text, tokens, 64);

    printf("Tokens: ");
    for (size_t i = 0; i < count; i++) {
        printf("%u ", tokens[i]);
    }
    printf("\n");

    int embed_dim = 4;
    embedding_layer *emb = embedding_new(256, embed_dim);

    tensor *vec = tensor_new(embed_dim, 1);
    embedding_forward(emb, tokens[0], vec);

    printf("Embedding for first token '%c':\n", text[0]);
    for (int i = 0; i < embed_dim; i++) {
        printf("  %f\n", vec->data[i]);
    }

    tensor_free(vec);
    embedding_free(emb);

    return 0;
}
