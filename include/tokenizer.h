#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

// Convert input string → token array (0-255 byte IDs)
// token_out must already be allocated with enough space.
size_t tokenize(const char *input, unsigned char *token_out, size_t max_tokens);

// Convert token array → output string (same bytes)
size_t detokenize(const unsigned char *tokens, size_t token_count, char *out, size_t max_out);

#endif
