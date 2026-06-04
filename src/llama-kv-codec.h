#pragma once

#include "llama.h"

#include <memory>

struct llama_kv_codec_i {
    virtual ~llama_kv_codec_i() = default;

    virtual llama_kv_cache_codec_type type() const = 0;
    virtual const char * name() const = 0;
};

using llama_kv_codec_ptr = std::unique_ptr<llama_kv_codec_i>;

llama_kv_codec_ptr llama_kv_codec_init(llama_kv_cache_codec_type type);