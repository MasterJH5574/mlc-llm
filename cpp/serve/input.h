/*!
 *  Copyright (c) 2023 by Contributors
 * \file input.h
 */
#ifndef MLC_LLM_SERVE_INPUT_H_
#define MLC_LLM_SERVE_INPUT_H_

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
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
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
  String text;

  static constexpr const char* _type_key = "mlc.serve.TextInput";
  TVM_DECLARE_BASE_OBJECT_INFO(TextInputNode, InputNode);
};

class TextInput : public Input {
 public:
  explicit TextInput(String text);

  TVM_DEFINE_OBJECT_REF_METHODS(TextInput, Input, TextInputNode);
};

/****************** TokenInputNode ******************/

/*! \brief The class of token input, containing a list of input tokens. */
class TokenInputNode : public InputNode {
 public:
  /*! \brief The input tokens. */
  ShapeTuple token_ids;

  static constexpr const char* _type_key = "mlc.serve.TokenInput";
  TVM_DECLARE_BASE_OBJECT_INFO(TokenInputNode, InputNode);
};

class TokenInput : public Input {
 public:
  explicit TokenInput(ShapeTuple token_ids);

  explicit TokenInput(std::vector<int32_t> token_ids);

  TVM_DEFINE_OBJECT_REF_METHODS(TokenInput, Input, TokenInputNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_INPUT_H_
