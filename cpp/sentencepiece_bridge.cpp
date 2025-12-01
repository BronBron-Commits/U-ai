#include <sentencepiece_processor.h>
#include <vector>
#include <string>
#include <cstring>

extern "C" {

struct SppModel {
    sentencepiece::SentencePieceProcessor processor;
};

void* spp_load(const char* path) {
    SppModel* m = new SppModel();
    auto status = m->processor.Load(path);
    if (!status.ok()) {
        delete m;
        return nullptr;
    }
    return m;
}

void spp_free(void* ptr) {
    delete static_cast<SppModel*>(ptr);
}

int spp_encode_ids(void* ptr,
                   const char* text,
                   int* out_ids,
                   int max_len)
{
    auto* m = static_cast<SppModel*>(ptr);
    std::vector<int> ids;

    auto status = m->processor.Encode(text, &ids);
    if (!status.ok()) return -1;

    int n = std::min(max_len, (int)ids.size());
    for (int i = 0; i < n; i++) {
        out_ids[i] = ids[i];
    }
    return n;
}

int spp_decode_ids(void* ptr,
                   const int* ids,
                   int len,
                   char* out_buf,
                   int buf_len)
{
    auto* m = static_cast<SppModel*>(ptr);
    std::vector<int> vec(ids, ids + len);

    std::string out;
    auto status = m->processor.Decode(vec, &out);
    if (!status.ok()) return -1;

    int n = std::min((int)out.size(), buf_len - 1);
    memcpy(out_buf, out.data(), n);
    out_buf[n] = '\0';
    return n;
}

int spp_vocab_size(void* ptr) {
    auto* m = static_cast<SppModel*>(ptr);
    return m->processor.GetPieceSize();
}

}
