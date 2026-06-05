#pragma once

#include "llama.h"

#include <memory>

struct llama_memory_params;

struct llama_kv_codec_i {
    virtual ~llama_kv_codec_i() = default;

    virtual llama_kv_cache_codec_type type() const = 0;
    virtual const char * name() const = 0;
    virtual ggml_type resolve_type_k(const llama_memory_params & params) const = 0;
    virtual ggml_type resolve_type_v(const llama_memory_params & params) const = 0;
    virtual llama_kv_cache_compressor_type resolve_compressor_k(const llama_memory_params & params) const = 0;
    virtual llama_kv_cache_compressor_type resolve_compressor_v(const llama_memory_params & params) const = 0;
    virtual bool has_runtime_encode() const {
        return false;
    }
};

using llama_kv_codec_ptr = std::unique_ptr<llama_kv_codec_i>;

llama_kv_codec_ptr llama_kv_codec_init(llama_kv_cache_codec_type type);