/*!
 *  Copyright (c) 2023 by Contributors
 * \file request.cc
 */

#include "request.h"

#include "input.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Request ******************/

TVM_REGISTER_OBJECT_TYPE(RequestNode);

Request::Request(Array<Input> inputs, String generation_cfg_json) {
  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->inputs = std::move(inputs);
  n->generation_cfg = GenerationConfig(generation_cfg_json);
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
