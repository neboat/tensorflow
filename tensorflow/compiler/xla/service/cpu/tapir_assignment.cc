/* Copyright 2019 Tao B. Schardl and the TensorFlow Authors. All
   Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/cpu/tapir_assignment.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"

namespace xla {
namespace cpu {

static const int64 L1_cache_size = 32LL << 10;  // 32KB L1 cache size.

class SimpleCostModel : public TapirCostModel {
 public:
  SimpleCostModel(const int64 max_parallelism,
                  const HloCostAnalysis::ShapeSizeFunction& shape_size)
      : max_parallelism_(max_parallelism), shape_size_(shape_size) {}
  ~SimpleCostModel() override {}

  int64 GetParallelTaskCount(HloInstruction* instruction) override {
    // Simple cost model based on hlo size and typical L2 cache size.
    const int64 instruction_cost = shape_size_(instruction->shape());
    // const int64 min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
    // // Return target parallel task count in [1, max_parallelism_].
    // return std::min(max_parallelism_,
    //                 std::max(int64{1}, instruction_cost / min_cost_per_thread));
    const int64 min_cost_per_thread = L1_cache_size;
    return std::max(int64{1}, instruction_cost / min_cost_per_thread);
  }

 private:
  const int64 max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
};

class DefaultCostModel : public TapirCostModel {
 public:
  DefaultCostModel(const int64 max_parallelism,
                   const HloCostAnalysis::ShapeSizeFunction& shape_size,
                   std::unique_ptr<HloCostAnalysis> cost_analysis)
      : max_parallelism_(max_parallelism),
        shape_size_(shape_size),
        cost_analysis_(std::move(cost_analysis)) {}
  ~DefaultCostModel() override {}

  int64 GetParallelTaskCount(HloInstruction* instruction) override {
    // Parameters for parallel task count computation.
    int64 instruction_cost;
    int64 min_cost_per_thread;
    int64 max_parallelism;
    // Calculate flops-to-bytes-ratio for 'instruction'.
    const int64 bytes_accessed =
        std::max(int64{1}, cost_analysis_->bytes_accessed(*instruction));
    const float flops_to_bytes_ratio =
        cost_analysis_->flop_count(*instruction) /
        static_cast<float>(bytes_accessed);
    // Check for I/O bound instructions.
    if (flops_to_bytes_ratio <= 1.0) {
      // Limit max parallelism for I/O bound instructions by assuming a
      // sub-linear scaling function (fit based on empirical benchmark results).
      // TODO(b/29630486) Develop system bandwidth model.
      max_parallelism =
          std::ceil(std::sqrt(tensorflow::port::NumSchedulableCPUs()));
      // Use shape size instruction cost and L2 cache size min per-thread cost.
      instruction_cost = shape_size_(instruction->shape());
      // min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
      min_cost_per_thread = L1_cache_size;
    } else {
      // Use max parallelism for compute bound instructions.
      max_parallelism = max_parallelism_;
      // Calculate the instruction cost in cycles.
      // TODO(b/29630486) Improve on this linear cost model.
      // Consider making 'min_cost_per_thread' be a function of the target
      // bandwidth limit for instructions with low arithmetic complexity.
      instruction_cost =
          1 * cost_analysis_->flop_count(*instruction) +
          2 * cost_analysis_->transcendental_count(*instruction) +
          10 * cost_analysis_->bytes_accessed(*instruction);
      // // Minimum per-thread cost is 100us of work on a 2GHz core.
      // min_cost_per_thread = 100000;
      min_cost_per_thread = 100;
    }
    // Return target parallel task count in [1, max_parallelism_].
    // return std::min(max_parallelism,
    //                 std::max(int64{1}, instruction_cost / min_cost_per_thread));
    return std::max(int64{1}, instruction_cost / min_cost_per_thread);
  }

 private:
  const int64 max_parallelism_;
  const HloCostAnalysis::ShapeSizeFunction shape_size_;
  const std::unique_ptr<HloCostAnalysis> cost_analysis_;
};

TapirAssignment::TapirAssignment(
    const int64 max_parallelism,
    const HloCostAnalysis::ShapeSizeFunction& shape_size, HloModule* module,
    const TargetMachineFeatures* target_machine_features)
    : target_machine_features_(*target_machine_features) {
  VLOG(1) << "TapirAssignment max_parallelism: " << max_parallelism;
  // Run cost analysis on 'module'.
  auto cost_analysis = absl::make_unique<HloCostAnalysis>(shape_size);
  HloComputation* computation = module->entry_computation();
  Status status = computation->root_instruction()->Accept(cost_analysis.get());
  if (status.ok()) {
    // Set default cost model based on 'cost_analysis'.
    cost_model_.reset(new DefaultCostModel(max_parallelism, shape_size,
                                           std::move(cost_analysis)));
  } else {
    // Fall back to a simple cost model based on hlo size and L2 cache size.
    // Note that HloCostAnalysis can returns an error status (likely because
    // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    cost_model_.reset(new SimpleCostModel(max_parallelism, shape_size));
  }
}

bool TapirAssignment::CanUseTapir(
    HloInstruction* instruction) {
  // Currently, we do not assign parallel tasks to instructions with at least
  // one of the following properties:
  // *) Internal threading (library calls to kConv, kDot, kFft, kCustomCall).
  // *) Emit custom loops (kSelectAndScatter).
  // *) Operations that are not thread safe (like infeed and rng).
  // *) Tuple-shaped.
  // TODO(b/27458679) Parallelize instructions which are skipped here.
  auto opcode = instruction->opcode();
  if (opcode == HloOpcode::kParameter || opcode == HloOpcode::kConstant ||
      opcode == HloOpcode::kCall || opcode == HloOpcode::kCustomCall ||
      opcode == HloOpcode::kDot || opcode == HloOpcode::kSelectAndScatter ||
      opcode == HloOpcode::kGetTupleElement || opcode == HloOpcode::kBitcast ||
      opcode == HloOpcode::kFft || opcode == HloOpcode::kInfeed ||
      opcode == HloOpcode::kOutfeed || opcode == HloOpcode::kRng ||
      opcode == HloOpcode::kSort ||
      (opcode == HloOpcode::kConvolution &&
       PotentiallyImplementedAsEigenConvolution(*instruction,
                                                target_machine_features_)) ||
      (opcode == HloOpcode::kFusion &&
       instruction->fusion_kind() != HloInstruction::FusionKind::kLoop) ||
      instruction->shape().IsTuple()) {
    return false;
  }
  // return (cost_model_->GetParallelTaskCount(instruction) > 1);
  return true;
}

StatusOr<bool> TapirAssigner::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "TapirAssigner ENTRY");
  XLA_VLOG_LINES(3, module->ToString());
  // Compute Tapir targets among all instructions in 'module'.
  HloToTapir hlo_to_tapir;
  int32 num_tapir_targets = ComputeTapirTargets(module, &hlo_to_tapir);

  XLA_VLOG_LINES(2, "TapirAssigner EXIT");
  XLA_VLOG_LINES(3, module->ToString());
  return (num_tapir_targets != 0);
}

int32 TapirAssigner::ComputeTapirTargets(HloModule* module,
					 HloToTapir* hlo_to_tapir) {
  TapirAssignment tapir_assignment(max_parallelism_,
				   shape_size_function_, module,
				   &target_machine_features_);
  int32 num_tapir_targets = 0;
  // Compute parallel task counts for all instructions in 'module'.
  for (auto* computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (auto* instruction : computation->instructions()) {
      if (tapir_assignment.CanUseTapir(instruction)) {
	instruction->setCodeGenUsingTapir(true);
	hlo_to_tapir->insert(instruction);
	num_tapir_targets++;
      }
    }
  }
  return num_tapir_targets;
}

}  // namespace cpu
}  // namespace xla
