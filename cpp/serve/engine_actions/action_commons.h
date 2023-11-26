/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/action_commons.h
 * \brief Common functions that may be used in multiple EngineActions.
 */
#ifndef MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
#define MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_

#include "../../tokenizers.h"
#include "../engine.h"
#include "../engine_state.h"
#include "../model.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Remove the given request from models.
 * \param estate The engine state to update after removal.
 * \param req_id The internal id of the request to remove.
 * \param models The models to remove the given request from.
 */
void RemoveRequestFromModel(EngineState estate, int req_id, Array<Model> models);

/*!
 * \brief The request post-processing after an engine action step.
 * It includes
 * - invoke the request function callback to return new generated tokens,
 * - update the engine state for finished requests.
 * \note This function may remove requests from the `running_queue`.
 * \param requests The requests to process.
 * \param estate The engine state.
 * \param models The models to remove the finished from.
 * \param request_stream_callback The request callback function.
 * \param max_single_sequence_length The max single sequence length to help decide
 * if a request is finished.
 */
void ActionStepPostProcess(Array<Request> requests, EngineState estate, Array<Model> models,
                           FRequestCallback request_stream_callback,
                           int max_single_sequence_length);

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_ENGINE_ACTIONS_ACTION_COMMONS_H_
