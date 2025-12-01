#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <cstring>

struct SppModel {
    sentencepiece::SentencePieceProcessor sp;
};

extern "C" {

// Load model
void* spp_load(const char* path) {
    SppModel* model = new SppModel();
    auto status = model->sp.Load(path);
    if (!status.ok()) {
        delete model;
        return nullptr;
    }
    return model;
}

// Free model
void spp_free(void* handle) {
    if (!handle) return;
    delete static_cast<SppModel*>(handle);
}

// Encode
int spp_encode_ids(void* handle, const char* text, int32_t* out_ids, int max_len) {
    if (!handle) return -1;
    SppModel* model = static_cast<SppModel*>(handle);

    std::vector<int> ids;
    auto status = model->sp.Encode(text, &ids);
    if (!status.ok()) return -1;

    int n = (int)ids.size();
    if (n > max_len) n = max_len;

    for (int i = 0; i < n; i++) {
        out_ids[i] = ids[i];
    }

    return n;
}

// Decode
int spp_decode_ids(void* handle, const int32_t* ids, int len, char* out_buf, int buf_size) {
    if (!handle) return -1;
    SppModel* model = static_cast<SppModel*>(handle);

    std::vector<int> vec(ids, ids + len);
    std::string out;

    auto status = model->sp.Decode(vec, &out);
    if (!status.ok()) return -1;

    if (out.size() + 1 > (size_t)buf_size)
        return -1;

    std::memcpy(out_buf, out.c_str(), out.size() + 1);
    return out.size();
}

// Vocab size
int spp_vocab_size(void* handle) {
    if (!handle) return -1;
    SppModel* model = static_cast<SppModel*>(handle);
    return model->sp.GetPieceSize();
}

} // extern "C"
