/*!
 *  Copyright (c) 2023 by Contributors
 * \file input.h
 */
#ifndef MLC_LLM_SERVE_INPUT_H_
#define MLC_LLM_SERVE_INPUT_H_

#include <tvm/node/reflection.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/****************** InputNode ******************/

/*! \brief The base class of multi-modality model inputs (text, tokens, embedding, etc). */
class InputNode : public Object {
 public:
  static constexpr const char* _type_key = "mlc.serve.Input";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(InputNode, Object);
};

class Input : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Input, ObjectRef, InputNode);
};

/****************** TextInputNode ******************/

/*! \brief The class of text input, containing a text string. */
class TextInputNode : public InputNode {
 public:
  /*! \brief The text input. */
  String prompt;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("prompt", &prompt); }

  bool SEqualReduce(const TextInputNode* other, tvm::SEqualReducer equal) const {
    return equal(prompt, other->prompt);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const { hash_reduce(prompt); }

  static constexpr const char* _type_key = "mlc.serve.TextInput";
  TVM_DECLARE_BASE_OBJECT_INFO(TextInputNode, InputNode);
};

class TextInput : public Input {
 public:
  explicit TextInput(String prompt);

  TVM_DEFINE_OBJECT_REF_METHODS(TextInput, Input, TextInputNode);
};

/****************** TokenInputNode ******************/

/*! \brief The class of token input, containing a list of input tokens. */
class TokenInputNode : public InputNode {
 public:
  /*! \brief The input tokens. */
  ShapeTuple token_ids;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("token_ids", &token_ids); }

  bool SEqualReduce(const TokenInputNode* other, tvm::SEqualReducer equal) const {
    return equal(token_ids, other->token_ids);
  }

  void SHashReduce(tvm::SHashReducer hash_reduce) const { hash_reduce(token_ids); }

  static constexpr const char* _type_key = "mlc.serve.TokenInput";
  TVM_DECLARE_BASE_OBJECT_INFO(TokenInputNode, InputNode);
};

class TokenInput : public Input {
 public:
  explicit TokenInput(ShapeTuple token_ids);

  explicit TokenInput(std::vector<int32_t> token_ids);

  explicit TokenInput(int32_t token_id);

  TVM_DEFINE_OBJECT_REF_METHODS(TokenInput, Input, TokenInputNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_INPUT_H_
