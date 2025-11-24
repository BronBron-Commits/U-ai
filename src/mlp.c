#include <stdlib.h>
#include "mlp.h"
#include "activation.h"
#include "tensor.h"

mlp *mlp_new(int input_size, int hidden_size) {
    mlp *m = malloc(sizeof(mlp));
    m->layer1 = linear_new(input_size, hidden_size);
    m->layer2 = linear_new(hidden_size, input_size);
    return m;
}

void mlp_free(mlp *m) {
    if (!m) return;
    linear_free(m->layer1);
    linear_free(m->layer2);
    free(m);
}

void mlp_forward(const mlp *m, const tensor *input, tensor *out) {
    tensor *hidden = tensor_new(m->layer1->out_features, 1);

    linear_forward(m->layer1, input, hidden);
    gelu(hidden);
    linear_forward(m->layer2, hidden, out);

    tensor_free(hidden);
}
