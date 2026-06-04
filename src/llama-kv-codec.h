#pragma once

#include <memory>

enum llama_kv_cache_codec_type {
    LLAMA_KV_CACHE_CODEC_TYPE_LEGACY = 0,
    LLAMA_KV_CACHE_CODEC_TYPE_COUNT,
};

const char * llama_kv_cache_codec_type_name(llama_kv_cache_codec_type type);

struct llama_kv_codec_i {
    virtual ~llama_kv_codec_i() = default;

    virtual llama_kv_cache_codec_type type() const = 0;
    virtual const char * name() const = 0;
};

using llama_kv_codec_ptr = std::unique_ptr<llama_kv_codec_i>;

llama_kv_codec_ptr llama_kv_codec_init(llama_kv_cache_codec_type type);