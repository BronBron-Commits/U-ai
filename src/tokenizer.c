#include <stdio.h>
#include <stdint.h>
#include <string.h>

static int dummy_tokenizer_instance = 123;

void* spp_load(const char *path) {
    if (path) {
        printf("[spp_load] loading: %s\n", path);
    } else {
        printf("[spp_load] received null path\n");
    }
    return &dummy_tokenizer_instance;
}

int spp_encode_ids(void* handle, const char *text, int32_t *out_ids, int32_t max_len) {
    (void)handle;
    *out_ids = 42;
    printf("[spp_encode_ids] encoding: %s\n", text);
    return 1;
}

int spp_decode_ids(void* handle, const int32_t *ids, int32_t len, char *out_text, int32_t max_len) {
    (void)handle;
    (void)len;
    snprintf(out_text, max_len, "[decoded %d]", ids[0]);
    printf("[spp_decode_ids] decoding id: %d\n", ids[0]);
    return strlen(out_text);
}

void spp_free(void* handle) {
    (void)handle;
    printf("[spp_free] cleanup\n");
}

int spp_vocab_size(void* handle) {
    (void)handle;
    return 50000;
}
