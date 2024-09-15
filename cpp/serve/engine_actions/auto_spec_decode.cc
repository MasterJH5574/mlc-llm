/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/auto_spec_decode.cc
 */

#include <tvm/runtime/nvtx.h>

#include <numeric>

#include "../config.h"
#include "action.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that first makes a decision on whether to run speculative
 * decoding or normal mode batch decode, and then runs the selected actions.
 */
class AutoSpecDecodeActionObj : public EngineActionObj {
 public:
  explicit AutoSpecDecodeActionObj(Array<EngineAction> spec_decode_actions,
                                   Array<EngineAction> batch_decode_actions,
                                   EngineConfig engine_config)
      : spec_decode_actions_(std::move(spec_decode_actions)),
        batch_decode_actions_(std::move(batch_decode_actions)),
        engine_config_(std::move(engine_config)) {}

  Array<Request> Step(EngineState estate) final {
    int num_running_rsentries = estate->GetRunningRequestStateEntries().size();
    if (num_running_rsentries == 0) {
      return {};
    }

    // Calculate the draft length to use for the next round decode.
    // LOG(INFO) << "batch size = " << num_running_rsentries;
    estate->spec_draft_length = CalculateDraftLength(estate, num_running_rsentries);
    ICHECK_GE(estate->spec_draft_length, 0);
    Array<Request> processed_requests;
    // Use speculative decoding when the computed draft length is positive.
    // Otherwise use normal mode batch decode.
    Array<EngineAction> actions =
        estate->spec_draft_length > 0 ? spec_decode_actions_ : batch_decode_actions_;
    for (EngineAction action : actions) {
      processed_requests = action->Step(estate);
    }

    // Reset the draft length.
    estate->spec_draft_length = 0;
    return processed_requests;
  }

 private:
  int CalculateDraftLength(EngineState estate, int num_running_rsentries) {
    if (num_running_rsentries >= kTokenBudget ||
        num_running_rsentries >= EngineMetrics::kEndFineGrainedTrackingBatchSize) {
      // The running request entries are more than the token budget,
      // in which case we always run normal mode batch decode.
      // LOG(INFO) << "exceeding budget. Draft length 0";
      return 0;
    }

    // Collect the draft length candidates. We consider three candidates:
    // 1. ceildiv(budget, num_running_rsentries),
    // 2. floordiv(budget, num_running_rsentries),
    // 3. 0, where we don't activate speculative decode.
    std::vector<int> draft_length_candidates;
    draft_length_candidates.reserve(3);
    draft_length_candidates.push_back(
        (kTokenBudget + num_running_rsentries - 1) / num_running_rsentries - 1);
    if (kTokenBudget % num_running_rsentries != 0) {
      draft_length_candidates.push_back(kTokenBudget / num_running_rsentries - 1);
    }
    if (draft_length_candidates.back() != 0) {
      draft_length_candidates.push_back(0);
    }

    // Check the validity of the draft length candidates.
    // Pick the candidate that has the lowest estimated per-token cost.
    int draft_length_decision = -1;
    double min_cost = std::numeric_limits<double>::max();
    for (int draft_length_candidate : draft_length_candidates) {
      int effective_batch_size = num_running_rsentries * (draft_length_candidate + 1);
      if (effective_batch_size > engine_config_->max_num_sequence ||
          effective_batch_size >= EngineMetrics::kEndFineGrainedTrackingBatchSize) {
        // Discard the candidate when using it will cause the effective batch size
        // exceeding the maximum allowed batch size.
        continue;
      }

      // Estimate the per-token time cost when using the draft length.
      double cost;
      if (draft_length_candidate != 0) {
        // LOG(INFO) << "query draft batch size " << num_running_rsentries << ", verify length "
                  // << effective_batch_size;
        cost = EstimateSpecDecodeCost(
            draft_length_candidate, estate->metrics.draft_time_by_batch_size[num_running_rsentries],
            estate->metrics.verify_time_by_batch_size[effective_batch_size],
            estate->metrics.spec_decode);
      } else {
        // LOG(INFO) << "query decode batch size " << num_running_rsentries;
        cost = EstimateBatchDecodeCost(
            estate->metrics.decode_time_by_batch_size[num_running_rsentries]);
      }
      // LOG(INFO) << "draft length " << draft_length_candidate << " has cost " << cost;
      if (cost < min_cost) {
        min_cost = cost;
        draft_length_decision = draft_length_candidate;
      }
    }
    ICHECK_NE(draft_length_decision, -1);
    // LOG(INFO) << "decision = " << draft_length_decision;
    return draft_length_decision;
  }

  double EstimateSpecDecodeCost(int draft_length, TimeCost draft_cost_per_step,
                                TimeCost verification_cost, SpecDecodeMetrics spec_decode_metrics) {
    if (draft_cost_per_step.count == 0 || verification_cost.count == 0) {
      // Return zero cost when there lacks statistics, in order to encourage exploration.
      return 0.0;
    }
    double total_draft_cost = draft_cost_per_step.sum / draft_cost_per_step.count * draft_length;
    double verify_cost = verification_cost.sum / verification_cost.count;

    // Calculate the avg number of accepted tokens when using the given draft length.
    int sum_num_acc_tokens = 0;
    int sum_num_draft_tokens = 0;
    for (int i = 0;
         i < std::min(draft_length + 1, static_cast<int>(spec_decode_metrics.accept_count.size()));
         ++i) {
      sum_num_acc_tokens += spec_decode_metrics.accept_count[i];
      sum_num_draft_tokens += spec_decode_metrics.draft_count[i];
    }
    if (sum_num_draft_tokens == 0) {
      // Return zero cost when there lacks statistics, in order to encourage exploration.
      return 0.0;
    }
    if (sum_num_acc_tokens == 0) {
      // Return inf if the average number of accepted tokens under this draft length is 0,
      // which means the per-token cost is infinity.
      return std::numeric_limits<double>::max();
    }
    double avg_num_accepted_tokens =
        1.0 * (draft_length + 1) * sum_num_acc_tokens / sum_num_draft_tokens;

    // Calculate the mean cost per token.
    return (total_draft_cost + verify_cost) / avg_num_accepted_tokens;
  }

  double EstimateBatchDecodeCost(TimeCost batch_decode_cost) {
    return batch_decode_cost.count > 0 ? batch_decode_cost.sum / batch_decode_cost.count : 0.0;
  }

  /*! \brief The fixed speculative decoding token budget. */
  static constexpr const int kTokenBudget = 64;
  /*! \brief The speculative decode actions. */
  Array<EngineAction> spec_decode_actions_;
  /*! \brief The normal mode decode actions. */
  Array<EngineAction> batch_decode_actions_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
};

EngineAction EngineAction::AutoSpecDecode(std::vector<EngineAction> spec_decode_actions_,
                                          std::vector<EngineAction> batch_decode_actions_,
                                          EngineConfig engine_config) {
  return EngineAction(make_object<AutoSpecDecodeActionObj>(
      Array<EngineAction>(spec_decode_actions_), Array<EngineAction>(batch_decode_actions_),
      std::move(engine_config)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
