#include <stdlib.h>
#include "init.h"

float rand_uniform(float a, float b) {
    float r = (float) rand() / (float) RAND_MAX;
    return a + (b - a) * r;
}
