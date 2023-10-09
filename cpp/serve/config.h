/*!
 *  Copyright (c) 2023 by Contributors
 * \file config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** Sampling config ******************/

/*! \brief The sampling configuration of a request. */
class SamplingParamsNode : public Object {
 public:
  double temperature = 0.8;
  double top_p = 0.95;
  double repetition_penalty = 1.0;

  int max_generation_length = 128;
  Array<String> stop_strs;

  static constexpr const char* _type_key = "mlc.serve.SamplingParams";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(SamplingParamsNode, Object);
};

class SamplingParams : public ObjectRef {
 public:
  explicit SamplingParams(String config_json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(SamplingParams, ObjectRef, SamplingParamsNode);
};

/****************** KV Cache config ******************/

/*! \brief The sampling configuration of a request. */
class KVCacheConfigNode : public Object {
 public:
  int page_size;
  int max_num_sequence;
  int max_total_sequence_length;

  static constexpr const char* _type_key = "mlc.serve.KVCacheConfig";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(KVCacheConfigNode, Object);
};

class KVCacheConfig : public ObjectRef {
 public:
  explicit KVCacheConfig(int page_size, int max_num_sequence, int max_total_sequence_length);

  explicit KVCacheConfig(const std::string& config_str, int max_single_sequence_length);

  TVM_DEFINE_OBJECT_REF_METHODS(KVCacheConfig, ObjectRef, KVCacheConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_CONFIG_H_
