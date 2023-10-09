/*!
 *  Copyright (c) 2023 by Contributors
 * \file config.h
 */
#ifndef MLC_LLM_SERVE_CONFIG_H_
#define MLC_LLM_SERVE_CONFIG_H_

#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** Sampling config ******************/

/*! \brief The sampling configuration of a request. */
class SampleConfigNode : public Object {
 public:
  double temperature = 0.8;
  double top_p = 0.95;
  double repetition_penalty = 1.0;

  int max_generation_length = 128;
  Array<String> stop_strs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("temperature", &temperature);
    v->Visit("top_p", &top_p);
    v->Visit("repetition_penalty", &repetition_penalty);
    v->Visit("max_generation_length", &max_generation_length);
    v->Visit("stop_strs", &stop_strs);
  }

  bool SEqualReduce(const SampleConfigNode* other, tvm::SEqualReducer equal) const {
    return equal(temperature, other->temperature) && equal(top_p, other->top_p) &&
           equal(repetition_penalty, other->repetition_penalty) &&
           equal(max_generation_length, other->max_generation_length) &&
           equal(stop_strs, other->stop_strs);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(temperature);
    hash_reduce(top_p);
    hash_reduce(repetition_penalty);
    hash_reduce(max_generation_length);
    hash_reduce(stop_strs);
  }

  static constexpr const char* _type_key = "mlc.serve.SampleConfig";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(SampleConfigNode, Object);
};

class SampleConfig : public ObjectRef {
 public:
  explicit SampleConfig(String config_json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(SampleConfig, ObjectRef, SampleConfigNode);
};

/****************** KV Cache config ******************/

/*! \brief The sampling configuration of a request. */
class KVCacheConfigNode : public Object {
 public:
  int page_size;
  int max_num_sequence;
  int max_total_sequence_length;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("page_size", &page_size);
    v->Visit("max_num_sequence", &max_num_sequence);
    v->Visit("max_total_sequence_length", &max_total_sequence_length);
  }

  bool SEqualReduce(const KVCacheConfigNode* other, tvm::SEqualReducer equal) const {
    return equal(page_size, other->page_size) && equal(max_num_sequence, other->max_num_sequence) &&
           equal(max_total_sequence_length, other->max_total_sequence_length);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(page_size);
    hash_reduce(max_num_sequence);
    hash_reduce(max_total_sequence_length);
  }

  static constexpr const char* _type_key = "mlc.serve.KVCacheConfig";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
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
