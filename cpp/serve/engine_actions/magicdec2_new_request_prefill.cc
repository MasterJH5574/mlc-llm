/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/new_request_prefill.cc
 */

#include "../sampler/sampler.h"
#include "batch_prefill_base.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that prefills requests in the `waiting_queue` of
 * the engine state.
 */
class MagicDec2NewRequestPrefillActionObj : public BatchPrefillBaseActionObj {
 public:
  explicit MagicDec2NewRequestPrefillActionObj(Array<Model> models, LogitProcessor logit_processor,
                                               Sampler sampler,
                                               std::vector<ModelWorkspace> model_workspaces,
                                               EngineConfig engine_config,
                                               std::vector<picojson::object> model_configs,
                                               Optional<EventTraceRecorder> trace_recorder)
      : BatchPrefillBaseActionObj(std::move(models), std::move(engine_config),
                                  std::move(model_configs), std::move(trace_recorder)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        model_workspaces_(std::move(model_workspaces)) {}

  Array<Request> Step(EngineState estate) final {
    // - Find the requests in `waiting_queue` that can prefill in this step.
    std::vector<PrefillInput> prefill_inputs;
    {
      NVTXScopedRange nvtx_scope("MagicDec2NewRequestPrefill getting requests");
      prefill_inputs = GetRequestStateEntriesToPrefill(estate);
      if (prefill_inputs.empty()) {
        return {};
      }
    }

    CHECK_EQ(prefill_inputs.size(), 1);
    {
      NVTXScopedRange nvtx_scope("MagicDec2NewRequestPrefill matching prefix");
      MatchPrefixCache(estate, &prefill_inputs[0]);
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // - Update status of request states from pending to alive.
    Array<String> request_ids;
    std::vector<RequestState> rstates_of_entries;
    std::vector<RequestStateStatus> status_before_prefill;
    UpdateRequestToAlive(prefill_inputs, estate, &request_ids, &rstates_of_entries,
                         &status_before_prefill);

    // - Get embedding and run prefill for each model.
    int64_t request_internal_id;
    const RequestStateEntry& rsentry = prefill_inputs[0].rsentry;
    RequestModelState mstate = rsentry->mstates[0];
    auto [input_data, input_length] =
        ChunkPrefillInputData(mstate, prefill_inputs[0].max_prefill_length);
    mstate->num_prefilled_tokens += input_length;

    ICHECK(mstate->draft_output_tokens.empty());
    ICHECK(mstate->draft_token_slots.empty());
    if (status_before_prefill[0] == RequestStateStatus::kPending &&
        !estate->prefix_cache->HasSequence(mstate->internal_id)) {
      // Add the sequence to the model, or fork the sequence from its parent.
      // If the sequence is already in prefix cache, it has also been added/forked in the
      // KVCache.
      CHECK(rsentry->parent_idx == -1);
      models_[0]->AddNewSequence(mstate->internal_id);
    }
    request_internal_id = mstate->internal_id;
    CHECK_EQ(input_data.size(), 1);
    mstate->prefilled_inputs.push_back(input_data[0]);
    ObjectRef embeddings = input_data[0]->GetEmbedding(models_[0],
                                                       /*dst=*/nullptr,
                                                       /*offset=*/0);

    RECORD_EVENT(trace_recorder_, request_ids, "start prefill");
    NDArray logits;
    if (mstate->inputs.empty()) {
      // There is no remaining input, which means the current input chunk should have
      // length same as the observation window size.
      CHECK_EQ(input_length, kObservationWindowSize);
      LOG(INFO) << "prefill request " << request_internal_id << " for last chunk "
                << kObservationWindowSize;
      logits = models_[0]->PrefillSnapKV(embeddings, request_internal_id, input_length,
                                         mstate->num_prefilled_tokens);
    } else {
      logits = models_[0]->BatchPrefill(embeddings, {request_internal_id}, {input_length});
    }
    RECORD_EVENT(trace_recorder_, request_ids, "finish prefill");
    ICHECK_EQ(logits->ndim, 3);
    ICHECK_EQ(logits->shape[0], 1);
    ICHECK_EQ(logits->shape[1], 1);

    // - Update logits.
    Array<GenerationConfig> generation_cfg{prefill_inputs[0].rsentry->request->generation_cfg};
    Array<RequestModelState> mstates_for_logitproc{prefill_inputs[0].rsentry->mstates[0]};
    logits = logits.CreateView({1, logits->shape[2]}, logits->dtype);
    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, mstates_for_logitproc,
                                          request_ids);

    // - Compute probability distributions.
    NDArray probs_on_device =
        logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

    // - Commit the prefix cache changes from previous round of action.
    // Note: we commit prefix cache changes here to overlap this commit with the GPU execution.
    estate->prefix_cache->CommitSequenceExtention();

    // - Sample tokens.
    //   For rsentries which have children, sample
    //   one token for each rstate that is depending.
    //   Otherwise, sample a token for the current rstate.
    std::vector<int> sample_indices;
    std::vector<RequestStateEntry> rsentries_for_sample;
    std::vector<RandomGenerator*> rngs;
    std::vector<bool> rsentry_activated;
    sample_indices.reserve(1);
    rsentries_for_sample.reserve(1);
    rngs.reserve(1);
    rsentry_activated.reserve(1);
    request_ids.clear();
    generation_cfg.clear();
    if (rsentry->mstates[0]->inputs.empty()) {
      CHECK_EQ(prefill_inputs[0].num_child_to_activate, 0);
      CHECK(rsentry->child_indices.empty());
      sample_indices.push_back(0);
      rsentries_for_sample.push_back(rsentry);
      request_ids.push_back(rsentry->request->id);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
      rsentry_activated.push_back(true);
    }

    NDArray renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
        renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
    ICHECK_EQ(sample_results.size(), rsentries_for_sample.size());

    // - Update the committed tokens of states.
    // - If a request is first-time prefilled, set the prefill finish time.
    UpdateRequestStateEntriesWithSampleResults(rsentries_for_sample, rsentry_activated,
                                               sample_results);

    auto tend = std::chrono::high_resolution_clock::now();
    estate->metrics.engine_prefill_time_sum += static_cast<double>((tend - tstart).count()) / 1e9;

    std::vector<Request> processed_requests =
        RemoveProcessedRequests(prefill_inputs, estate, rstates_of_entries);
    estate->running_rsentries_changed = true;
    return processed_requests;
  }

 private:
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief Workspace of each model. */
  std::vector<ModelWorkspace> model_workspaces_;
  static constexpr const int kObservationWindowSize = 32;

  /*!
   * \brief Match the request state entry with prefix cache, to skip prefilling common prefix
   * tokens. If the request state entry is not added to KVCache yet, this method will add/fork the
   * request in the KVCache, depending on the matching result from prefix cache.
   * \param estate The engine state.
   * \param[in, out] input The prefill input to be matched and updated.
   */
  void MatchPrefixCache(EngineState estate, PrefillInput* input) final {
    RequestStateEntry rsentry = input->rsentry;
    if (estate->prefix_cache->Mode() == PrefixCacheMode::kDisable) {
      return;
    }
    if (rsentry->parent_idx == -1 && rsentry->status == RequestStateStatus::kPending &&
        !estate->prefix_cache->HasSequence(rsentry->mstates[0]->internal_id)) {
      std::vector<int32_t> tokens = GetConcatPrefillInputData(rsentry->mstates[0]);
      if (tokens.empty()) {
        // If the RequestStateEntry is of empty input data, or not fully tokenized, do nothing
        // and return.
        return;
      }
      PrefixCacheMatchedResult result = estate->prefix_cache->InsertSequence(
          rsentry->mstates[0]->internal_id, tokens, models_[0]->GetSlidingWindowSize(),
          models_[0]->GetAttentionSinkSize());

      if (result.prefilled_offset == 0) {
        // Add new sequence
        CHECK_EQ(result.forked_seq_id, -1);
        CHECK_EQ(result.reused_seq_id, -1);
        CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
        for (Model model : models_) {
          model->AddNewSequence(rsentry->mstates[0]->internal_id);
          // Enable sliding window for the sequence if it is not a parent.
          if (rsentry->child_indices.empty()) {
            model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
          }
        }
      } else {
        if (result.forked_seq_id != -1) {
          CHECK_EQ(result.reused_seq_id, -1);
          CHECK_EQ(result.reused_seq_pop_last_tokens, 0);
          // Fork from active sequence
          for (Model model : models_) {
            model->ForkSequence(result.forked_seq_id, rsentry->mstates[0]->internal_id,
                                result.prefilled_offset);
            // Enable sliding window for the sequence if it is not a parent.
            if (rsentry->child_indices.empty()) {
              model->EnableSlidingWindowForSeq(rsentry->mstates[0]->internal_id);
            }
          }
        } else {
          // Reuse recycling sequence
          CHECK_EQ(result.forked_seq_id, -1);
          estate->id_manager.RecycleId(rsentry->mstates[0]->internal_id);
          for (int i = 0; i < rsentry->mstates.size(); ++i) {
            rsentry->mstates[i]->internal_id = result.reused_seq_id;
          }
          if (result.reused_seq_pop_last_tokens > 0) {
            for (Model model : models_) {
              model->PopNFromKVCache(rsentry->mstates[0]->internal_id,
                                     result.reused_seq_pop_last_tokens);
            }
          }
        }
      }
      // Pop matched prefix
      if (result.prefilled_offset) {
        for (int i = 0; i < rsentry->mstates.size(); ++i) {
          PopPrefillInputData(rsentry->mstates[i], result.prefilled_offset);
        }
      }
      // Update max prefill length
      input->max_prefill_length =
          std::min(input->max_prefill_length, rsentry->mstates[0]->GetInputLength());
    }
  }

  std::vector<PrefillInput> GetRequestStateEntriesToPrefill(EngineState estate) {
    // Preempt request state entries when decode cannot apply.
    const std::vector<RequestStateEntry>* running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchDecode getting requests");
      running_rsentries = &estate->GetRunningRequestStateEntries();
      if (!(running_rsentries->size() <= models_[0]->GetNumAvailablePages())) {
        // Even the decode cannot be performed.
        // As a result, directly return without doing prefill.
        return {};
      }
    }

    if (estate->waiting_queue.empty()) {
      // No request to prefill.
      return {};
    }

    ICHECK_EQ(models_.size(), 1);
    PrefillInput prefill_input;

    int num_decode_inputs = 0;

    // - Try to prefill pending requests.
    int total_input_length = 0;
    int total_required_pages = 0;
    int num_available_pages;
    int num_running_rsentries = 0;
    int current_total_seq_len;
    {
      NVTXScopedRange nvtx_scope("KV cache GetNumAvailablePages");
      num_available_pages = models_[0]->GetNumAvailablePages();
    }
    {
      NVTXScopedRange nvtx_scope("KV cache GetCurrentTotalSequenceLength");
      current_total_seq_len = models_[0]->GetCurrentTotalSequenceLength();
    }

    Request request = estate->waiting_queue[0];
    {
      NVTXScopedRange nvtx_scope("Process request " + request->id);
      RequestState rstate = estate->GetRequestState(request);
      bool prefill_stops = false;
      CHECK_EQ(rstate->entries.size(), 1);
      RequestStateEntry rsentry = rstate->entries[0];
      // A request state entry can be prefilled only when:
      // - it has inputs, and
      // - it has no parent or its parent is alive and has no remaining input.
      CHECK(!(rsentry->mstates[0]->inputs.empty() ||
              (rsentry->parent_idx != -1 &&
               (rstate->entries[rsentry->parent_idx]->status == RequestStateStatus::kPending ||
                !rstate->entries[rsentry->parent_idx]->mstates[0]->inputs.empty()))));

      int input_length = rsentry->mstates[0]->GetInputLength();
      int num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                              engine_config_->kv_cache_page_size;
      bool sliding_window_enabled = false;

      // The input length is expectede to be at least as long as the observation window size.
      CHECK_GE(input_length, kObservationWindowSize);

      // - Attempt 1. Check if the entire request state entry can fit for prefill.
      // We only do this when the remaining length is the same as the observation window size.
      if (input_length == kObservationWindowSize) {
        NVTXScopedRange nvtx_scope("Attempt 1");
        CHECK(rsentry->child_indices.empty());
        CHECK(CanPrefill(estate, 1, input_length, num_require_pages, num_available_pages,
                         current_total_seq_len, num_running_rsentries, kv_state_kind_,
                         sliding_window_enabled));
        return {{rsentry, input_length, 0, /*is_decode=*/false}};
      }

      // We reserve the last chunk to be the observation window.
      input_length -= kObservationWindowSize;

      // - Attempt 2. Check if the request state entry can partially fit by input chunking.
      input_length = std::min(input_length, static_cast<int>(engine_config_->prefill_chunk_size));
      num_require_pages = (input_length + engine_config_->kv_cache_page_size - 1) /
                          engine_config_->kv_cache_page_size;

      {
        NVTXScopedRange nvtx_scope("Attempt 2");
        total_input_length += input_length;
        total_required_pages += num_require_pages;
        CHECK(CanPrefill(estate, 0, input_length, num_require_pages, num_available_pages,
                         current_total_seq_len, num_running_rsentries, kv_state_kind_,
                         sliding_window_enabled));
        return {{rsentry, input_length, 0, /*is_decode=*/false}};
      }
    }
  }
};

EngineAction EngineAction::MagicDec2NewRequestPrefill(
    Array<Model> models, LogitProcessor logit_processor, Sampler sampler,
    std::vector<ModelWorkspace> model_workspaces, EngineConfig engine_config,
    std::vector<picojson::object> model_configs, Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<MagicDec2NewRequestPrefillActionObj>(
      std::move(models), std::move(logit_processor), std::move(sampler),
      std::move(model_workspaces), std::move(engine_config), std::move(model_configs),
      std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
