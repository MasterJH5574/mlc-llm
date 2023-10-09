/*!
 *  Copyright (c) 2023 by Contributors
 * \file input.cc
 */
#include "input.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Inputs ******************/

TVM_REGISTER_OBJECT_TYPE(InputNode);

TVM_REGISTER_OBJECT_TYPE(TextInputNode);

TextInput::TextInput(String text) {
  ObjectPtr<TextInputNode> n = make_object<TextInputNode>();
  n->text = std::move(text);
  data_ = std::move(n);
}

TVM_REGISTER_OBJECT_TYPE(TokenInputNode);

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

}  // namespace serve
}  // namespace llm
}  // namespace mlc
