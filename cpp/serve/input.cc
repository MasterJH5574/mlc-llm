/*!
 *  Copyright (c) 2023 by Contributors
 * \file input.cc
 */
#include "input.h"

#include <tvm/node/reflection.h>

namespace mlc {
namespace llm {
namespace serve {

/****************** Inputs ******************/

// TVM_REGISTER_NODE_TYPE(TextNode);

TextInput::TextInput(String prompt) {
  ObjectPtr<TextInputNode> n = make_object<TextInputNode>();
  n->prompt = std::move(prompt);
  data_ = std::move(n);
}

TokenInput::TokenInput(ShapeTuple token_ids) {
  ObjectPtr<TokenInputNode> n = make_object<TokenInputNode>();
  n->token_ids = std::move(token_ids);
  data_ = std::move(n);
}

TokenInput::TokenInput(std::vector<int32_t> token_ids) {
  ObjectPtr<TokenInputNode> n = make_object<TokenInputNode>();
  n->token_ids = ShapeTuple(token_ids.begin(), token_ids.end());
  data_ = std::move(n);
}

TokenInput::TokenInput(int32_t token_id) {
  ObjectPtr<TokenInputNode> n = make_object<TokenInputNode>();
  n->token_ids = ShapeTuple{token_id};
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
