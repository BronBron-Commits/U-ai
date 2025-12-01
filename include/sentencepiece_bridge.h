#ifndef SENTENCEPIECE_BRIDGE_H
#define SENTENCEPIECE_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load a SentencePiece model (.model / .spm)
void* spp_load(const char* model_path);

// Encode text -> token IDs
int spp_encode_ids(void* ptr, const char* text, int* out_ids, int max_ids);

// Decode token IDs -> text
int spp_decode_ids(void* ptr, const int32_t* ids, int len, char* out_buf, int buf_len);

// Return vocab size
int spp_vocab_size(void* ptr);

// Free the model
void spp_free(void* ptr);

#ifdef __cplusplus
}
#endif

#endif // SENTENCEPIECE_BRIDGE_H
