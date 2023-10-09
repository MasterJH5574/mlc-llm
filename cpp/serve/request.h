/*!
 *  Copyright (c) 2023 by Contributors
 * \file request.h
 * \brief Implementation of llm chat.
 */
#ifndef MLC_LLM_SERVE_REQUEST_H_
#define MLC_LLM_SERVE_REQUEST_H_

#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include "config.h"
#include "input.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** Request ******************/

/*!
 * \brief The user submitted text-generation request, which contains
 * a list of multi-modal inputs and a set of sampling configuration
 * parameters.
 */
class RequestNode : public Object {
 public:
  Array<Input> inputs;
  /*! \brief JSON string containing temperature, top_p, repetition_penalty, max_gen_len, etc. */
  SampleConfig sampling_params;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("inputs", &inputs);
    v->Visit("sampling_params", &sampling_params);
  }

  bool SEqualReduce(const RequestNode* other, tvm::SEqualReducer equal) const {
    return equal(inputs, other->inputs) && equal(sampling_params, other->sampling_params);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    hash_reduce(inputs);
    hash_reduce(sampling_params);
  }

  static constexpr const char* _type_key = "mlc.serve.Request";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(RequestNode, Object);
};

class Request : public ObjectRef {
 public:
  explicit Request(Array<Input> inputs, String sampling_params_json);

  TVM_DEFINE_OBJECT_REF_METHODS(Request, ObjectRef, RequestNode);
};

/****************** Request State ******************/

/*!
 * \brief The state of a request with regard to some single model.
 * \details In MLC LLM, the serving engine may leverage multiple models
 * to fulfill a user generation request (e.g., use speculation decoding).
 * For each request, we isolate its states (e.g. the generated tokens)
 * on each model. This is to say, we use RequestModelState to store
 * the state of a user request on a single model (rather than all models).
 */
class RequestModelStateNode : public Object {
 public:
  /*!
   * \brief The corresponding request id of this state.
   * It is the **physical index** of the request in the running request queue.
   * If the request is on hold (not in the running queue), the request id
   * should be -1.
   */
  int request_id = -1;
  /*! \brief The corresponding model id of this state. */
  int model_id = -1;
  /*!
   * \brief The committed generated token ids. A token is "committed"
   * means it will no longer be updated (or changed).
   */
  std::vector<int32_t> committed_tokens;
  /*! \brief The list of inputs yet for the model to prefill. */
  Array<Input> input_to_prefill;
  /*!
   * \brief The draft generated token ids, which are usually generated
   * by "small" speculative models. These tokens will be fed to a "large"
   * model to determine the final result of speculation.
   */
  std::vector<int32_t> draft_output_tokens;
  /*!
   * \brief The probability distribution on each position in the
   * draft. We keep the distributions for stochastic sampling when merging
   * speculations from multiple models.
   */
  std::vector<std::vector<float>> draft_output_prob_dist;
  /*!
   * \brief The probability of the sampled token on each position in the
   * draft. We keep the probabilities for stochastic sampling when merging
   * speculations from multiple models.
   *
   * \note `draft_token_prob` can be inferred from `draft_tokens` and
   * `draft_prob_dist`, but we still keep it so that we can have option
   * choosing only to use one between them.
   */
  std::vector<float> draft_output_token_prob;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // Expose nothing for now.
  }

  bool SEqualReduce(const RequestModelStateNode* other, tvm::SEqualReducer equal) const {
    auto f_int_vector_equal = [](const std::vector<int32_t>& va, const std::vector<int32_t>& vb) {
      if (va.size() != vb.size()) return false;
      for (int i = 0; i < static_cast<int>(va.size()); ++i) {
        if (va[i] != vb[i]) return false;
      }
      return true;
    };
    auto f_float_vector_equal = [](const std::vector<float>& va, const std::vector<float>& vb) {
      if (va.size() != vb.size()) return false;
      for (int i = 0; i < static_cast<int>(va.size()); ++i) {
        if (va[i] != vb[i]) return false;
      }
      return true;
    };
    auto f_nested_float_vector_equal = [&f_float_vector_equal](
                                           const std::vector<std::vector<float>>& va,
                                           const std::vector<std::vector<float>>& vb) {
      if (va.size() != vb.size()) return false;
      for (int i = 0; i < static_cast<int>(va.size()); ++i) {
        if (!f_float_vector_equal(va[i], vb[i])) return false;
      }
      return true;
    };

    return equal(request_id, other->request_id) && equal(model_id, other->model_id) &&
           equal(input_to_prefill, other->input_to_prefill) &&
           f_int_vector_equal(committed_tokens, other->committed_tokens) &&
           f_int_vector_equal(draft_output_tokens, other->draft_output_tokens) &&
           f_float_vector_equal(draft_output_token_prob, other->draft_output_token_prob) &&
           f_nested_float_vector_equal(draft_output_prob_dist, other->draft_output_prob_dist);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const {
    auto f_hash_int_vector = [&hash_reduce](const std::vector<int32_t>& vec) {
      hash_reduce(vec.size());
      for (int32_t value : vec) {
        hash_reduce(value);
      }
    };
    auto f_hash_float_vector = [&hash_reduce](const std::vector<float>& vec) {
      hash_reduce(vec.size());
      for (float value : vec) {
        hash_reduce(value);
      }
    };
    auto f_hash_nested_float_vector =
        [&hash_reduce, &f_hash_float_vector](const std::vector<std::vector<float>>& vec) {
          hash_reduce(vec.size());
          for (const std::vector<float>& value : vec) {
            f_hash_float_vector(value);
          }
        };

    hash_reduce(request_id);
    hash_reduce(model_id);
    hash_reduce(input_to_prefill);
    f_hash_int_vector(committed_tokens);
    f_hash_int_vector(draft_output_tokens);
    f_hash_float_vector(draft_output_token_prob);
    f_hash_nested_float_vector(draft_output_prob_dist);
  }

  static constexpr const char* _type_key = "mlc.serve.RequestModelState";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(RequestModelStateNode, Object);
};

class RequestModelState : public ObjectRef {
 public:
  explicit RequestModelState(int model_id, Array<Input> input_to_prefill);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(RequestModelState, ObjectRef, RequestModelStateNode);
};

struct RequestState {
  /*!
   * \brief The state with regard to each model.
   * \sa RequestModelState
   */
  Array<RequestModelState> mstates;

  /*! \brief The decoded text string output. */
  std::string output = "";
  /*! \brief The boolean flag indicating if the process of this request is finished. */
  bool finished = false;

  explicit RequestState(int num_models, Array<Input> inputs) {
    mstates.reserve(num_models);
    for (int i = 0; i < num_models; ++i) {
      mstates.push_back(RequestModelState(i, inputs));
    }
  }
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_REQUEST_H_
