/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.h
 * \brief The header of serving engine in MLC LLM.
 */
#ifndef MLC_LLM_SERVE_ENGINE_H_
#define MLC_LLM_SERVE_ENGINE_H_

#include <tvm/runtime/packed_func.h>

#include "data.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

typedef TypedPackedFunc<void(String, TokenData, Optional<String>)> FRequestCallback;

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_H_
