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

Request::Request(Array<Input> inputs, String sampling_params_json) {
  ObjectPtr<RequestNode> n = make_object<RequestNode>();
  n->inputs = std::move(inputs);
  n->sampling_params = SamplingParams(sampling_params_json);
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
