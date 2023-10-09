/*!
 *  Copyright (c) 2023 by Contributors
 * \file request_state.cc
 */

#include "request_state.h"

#include "input.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** RequestModelState ******************/

TVM_REGISTER_OBJECT_TYPE(RequestModelStateNode);

RequestModelState::RequestModelState(int model_id, Array<Input> inputs) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->model_id = model_id;
  n->request_id = -1;
  n->inputs = std::move(inputs);
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
