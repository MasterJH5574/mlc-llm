/*!
 *  Copyright (c) 2023 by Contributors
 * \file request.cc
 */

#include "request.h"

#include <tvm/node/reflection.h>

#include "config.h"
#include "input.h"

namespace mlc {
namespace llm {
namespace serve {

/****************** Request ******************/

// TVM_REGISTER_NODE_TYPE(RequestNode);

Request::Request(Array<Input> inputs, String sampling_params_json) {
  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->inputs = std::move(inputs);
  n->sampling_params = SampleConfig(sampling_params_json);
  data_ = std::move(n);
}

/****************** RequestModelState ******************/

RequestModelState::RequestModelState(int model_id, Array<Input> input_to_prefill) {
  ObjectPtr<RequestModelStateNode> n = make_object<RequestModelStateNode>();
  n->model_id = model_id;
  n->request_id = -1;
  n->input_to_prefill = std::move(input_to_prefill);
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
