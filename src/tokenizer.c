#include "tokenizer.h"
#include <string.h>

// Straight byte-level tokenizer.
// Each char becomes a token in range 0-255.
size_t tokenize(const char *input, unsigned char *token_out, size_t max_tokens) {
    size_t len = strlen(input);
    if (len > max_tokens) len = max_tokens;

    for (size_t i = 0; i < len; i++) {
        token_out[i] = (unsigned char)input[i];
    }

    return len;
}

// Convert tokens back to bytes.
size_t detokenize(const unsigned char *tokens, size_t token_count, char *out, size_t max_out) {
    if (token_count > max_out - 1)
        token_count = max_out - 1;

    for (size_t i = 0; i < token_count; i++) {
        out[i] = (char)tokens[i];
    }

    out[token_count] = '\0';
    return token_count;
}
