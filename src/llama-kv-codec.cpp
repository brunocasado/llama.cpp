#include "llama-kv-codec.h"
#include "llama-memory.h"

namespace {

static ggml_type turboquant_default_type_k() {
    return GGML_TYPE_Q4_0;
}

static ggml_type turboquant_default_type_v() {
    return GGML_TYPE_Q4_0;
}

static llama_kv_cache_compressor_type turboquant_default_compressor_k(ggml_type type_k) {
    if (ggml_is_quantized(type_k)) {
        return LLAMA_KV_CACHE_COMPRESSOR_TYPE_HADAMARD;
    }

    return LLAMA_KV_CACHE_COMPRESSOR_TYPE_DIRECT;
}

static llama_kv_cache_compressor_type turboquant_default_compressor_v(ggml_type type_v) {
    if (ggml_is_quantized(type_v)) {
        return LLAMA_KV_CACHE_COMPRESSOR_TYPE_HADAMARD;
    }

    return LLAMA_KV_CACHE_COMPRESSOR_TYPE_DIRECT;
}

struct llama_kv_codec_legacy : public llama_kv_codec_i {
    llama_kv_cache_codec_type type() const override {
        return LLAMA_KV_CACHE_CODEC_TYPE_LEGACY;
    }

    const char * name() const override {
        return llama_kv_cache_codec_type_name(type());
    }

    ggml_type resolve_type_k(const llama_memory_params & params) const override {
        return params.type_k;
    }

    ggml_type resolve_type_v(const llama_memory_params & params) const override {
        return params.type_v;
    }

    llama_kv_cache_compressor_type resolve_compressor_k(const llama_memory_params &) const override {
        return LLAMA_KV_CACHE_COMPRESSOR_TYPE_DIRECT;
    }

    llama_kv_cache_compressor_type resolve_compressor_v(const llama_memory_params &) const override {
        return LLAMA_KV_CACHE_COMPRESSOR_TYPE_DIRECT;
    }
};

struct llama_kv_codec_turboquant : public llama_kv_codec_i {
    llama_kv_cache_codec_type type() const override {
        return LLAMA_KV_CACHE_CODEC_TYPE_TURBOQUANT;
    }

    const char * name() const override {
        return llama_kv_cache_codec_type_name(type());
    }

    ggml_type resolve_type_k(const llama_memory_params & params) const override {
        return params.type_k == GGML_TYPE_F16 ? turboquant_default_type_k() : params.type_k;
    }

    ggml_type resolve_type_v(const llama_memory_params & params) const override {
        return params.type_v == GGML_TYPE_F16 ? turboquant_default_type_v() : params.type_v;
    }

    llama_kv_cache_compressor_type resolve_compressor_k(const llama_memory_params & params) const override {
        const ggml_type type_k = resolve_type_k(params);
        if (params.kv_compressor_k != LLAMA_KV_CACHE_COMPRESSOR_TYPE_AUTO) {
            return params.kv_compressor_k;
        }

        return turboquant_default_compressor_k(type_k);
    }

    llama_kv_cache_compressor_type resolve_compressor_v(const llama_memory_params & params) const override {
        const ggml_type type_v = resolve_type_v(params);
        if (params.kv_compressor_v != LLAMA_KV_CACHE_COMPRESSOR_TYPE_AUTO) {
            return params.kv_compressor_v;
        }

        return turboquant_default_compressor_v(type_v);
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
