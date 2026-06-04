#include "llama-kv-codec.h"

namespace {

struct llama_kv_codec_legacy : public llama_kv_codec_i {
    llama_kv_cache_codec_type type() const override {
        return LLAMA_KV_CACHE_CODEC_TYPE_LEGACY;
    }

    const char * name() const override {
        return llama_kv_cache_codec_type_name(type());
    }

    bool uses_quantized_kv() const override {
        return false;
    }
};

struct llama_kv_codec_turboquant : public llama_kv_codec_i {
    llama_kv_cache_codec_type type() const override {
        return LLAMA_KV_CACHE_CODEC_TYPE_TURBOQUANT;
    }

    const char * name() const override {
        return llama_kv_cache_codec_type_name(type());
    }

    bool uses_quantized_kv() const override {
        return true;
    }
};

}

llama_kv_codec_ptr llama_kv_codec_init(llama_kv_cache_codec_type type) {
    switch (type) {
        case LLAMA_KV_CACHE_CODEC_TYPE_LEGACY:
            {
                return std::make_unique<llama_kv_codec_legacy>();
            }
        case LLAMA_KV_CACHE_CODEC_TYPE_TURBOQUANT:
            {
                return std::make_unique<llama_kv_codec_turboquant>();
            }
    }

    return nullptr;
}
