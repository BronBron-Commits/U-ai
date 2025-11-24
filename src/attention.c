#include <stdlib.h>
#include <math.h>
#include "attention.h"
#include "softmax.h"
#include "tensor.h"
#include "linear.h"

attention_head *attention_new(int embed_dim) {
    attention_head *att = malloc(sizeof(attention_head));
    att->embed_dim = embed_dim;

    att->Wq = linear_new(embed_dim, embed_dim);
    att->Wk = linear_new(embed_dim, embed_dim);
    att->Wv = linear_new(embed_dim, embed_dim);

    return att;
}

void attention_free(attention_head *att) {
    if (!att) return;

    linear_free(att->Wq);
    linear_free(att->Wk);
    linear_free(att->Wv);

    free(att);
}

void attention_forward(const attention_head *att, const tensor *x, tensor *out) {
    int d = att->embed_dim;

    // Q, K, V each shape: (d x 1)
    tensor *Q = tensor_new(d, 1);
    tensor *K = tensor_new(d, 1);
    tensor *V = tensor_new(d, 1);

    linear_forward(att->Wq, x, Q);
    linear_forward(att->Wk, x, K);
    linear_forward(att->Wv, x, V);

    // Score = Q dot K / sqrt(d)
    float score = 0.0f;
    for (int i = 0; i < d; i++) {
        score += Q->data[i] * K->data[i];
    }
    score /= sqrtf((float)d);

    // Convert score to a 1×1 tensor so we can softmax it
    tensor *score_t = tensor_new(1, 1);
    score_t->data[0] = score;

    // Softmax (for a single token, softmax(x) = 1, but this is correct form)
    softmax(score_t);

    float weight = score_t->data[0];

    // Output = weight * V
    for (int i = 0; i < d; i++) {
        out->data[i] = V->data[i] * weight;
    }

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(score_t);
}
