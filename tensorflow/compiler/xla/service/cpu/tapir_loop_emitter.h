/* Copyright 2019 Tao B. Schardl and the TensorFlow Authors.  All Rights
   Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TAPIR_LOOP_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TAPIR_LOOP_EMITTER_H_

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/loop_emitter.h"

namespace xla {
namespace cpu {

// TapirLoopEmitter emits a Tapir loop nest for the target loop.  Based on
// ParallelLoopEmitter.
class TapirLoopEmitter : public llvm_ir::LoopEmitter {
 public:
  // Constructs a TapirLoopEmitter which uses 'target_element_generator' to
  // generate elements, 'dynamic_loop_bounds' to set the loop bounds of the
  // most-major dimensions, and 'target_array.' shape to set the static loop
  // bounds for the most-minor dimensions.
  TapirLoopEmitter(const llvm_ir::ElementGenerator& target_element_generator,
                   const llvm_ir::IrArray& target_array,
                   llvm::IRBuilder<>* b);

  // TODO: Figure out if we can handle this loop with Tapir.
  //
  // // Constructs a LoopEmitter that emits one element into each of N separate
  // // arrays on each iteration of the loop.
  // //
  // // This is used for multi-output fusion.  target_element_generator must
  // // produce an LLVM struct with N elements.
  // TapirLoopEmitter(const ElementGenerator& target_element_generator,
  //                  absl::Span<const IrArray> target_arrays, llvm::IRBuilder<>* b);

  TapirLoopEmitter(const TapirLoopEmitter&) = delete;
  TapirLoopEmitter& operator=(const TapirLoopEmitter&) = delete;
  ~TapirLoopEmitter() override = default;

  std::vector<llvm_ir::IrArray::Index> EmitIndexAndSetExitBasicBlock(
      absl::string_view loop_name, llvm::Type* index_type) override;
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_TAPIR_LOOP_EMITTER_H_
