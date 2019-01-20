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
    // // Set default cost model based on 'cost_analysis'.
    // cost_model_.reset(new DefaultCostModel(max_parallelism, shape_size,
    //                                        std::move(cost_analysis)));
  } else {
    // // Fall back to a simple cost model based on hlo size and L2 cache size.
    // // Note that HloCostAnalysis can returns an error status (likely because
    // // HLOs like CustomCall are not yet implemented in the HloCostAnalysis).
    // cost_model_.reset(new SimpleCostModel(max_parallelism, shape_size));
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
