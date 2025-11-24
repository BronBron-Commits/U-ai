#include <stdio.h>
#include "tensor.h"
#include "mlp.h"

int main() {
    // Input vector of size 3
    tensor *x = tensor_new(3, 1);
    x->data[0] = 1.0f;
    x->data[1] = 2.0f;
    x->data[2] = 3.0f;

    // Create an MLP: 3 -> 6 -> 3
    mlp *m = mlp_new(3, 6);

    // Manually set some weights for now (random-ish)
    for (int i = 0; i < m->layer1->weight->rows * m->layer1->weight->cols; i++)
        m->layer1->weight->data[i] = 0.1f * (i + 1);

    for (int i = 0; i < m->layer2->weight->rows * m->layer2->weight->cols; i++)
        m->layer2->weight->data[i] = 0.05f * (i + 1);

    // Bias values
    for (int i = 0; i < m->layer1->bias->rows; i++)
        m->layer1->bias->data[i] = 0.01f * (i + 1);

    for (int i = 0; i < m->layer2->bias->rows; i++)
        m->layer2->bias->data[i] = -0.01f * (i + 1);

    // Output vector
    tensor *out = tensor_new(3, 1);

    mlp_forward(m, x, out);

    printf("MLP output:\n");
    printf("  y0 = %f\n", out->data[0]);
    printf("  y1 = %f\n", out->data[1]);
    printf("  y2 = %f\n", out->data[2]);

    tensor_free(x);
    tensor_free(out);
    mlp_free(m);

    return 0;
}
