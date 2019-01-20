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

#include "tensorflow/compiler/xla/service/cpu/tapir_loop_emitter.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
namespace cpu {

TapirLoopEmitter::TapirLoopEmitter(
    const llvm_ir::ElementGenerator& target_element_generator,
    const llvm_ir::IrArray& target_array, llvm::IRBuilder<>* b)
    : LoopEmitter(target_element_generator, target_array, b) {}


std::vector<llvm_ir::IrArray::Index>
TapirLoopEmitter::EmitIndexAndSetExitBasicBlock(absl::string_view loop_name,
						llvm::Type* index_type) {
  CHECK_NE(index_type, nullptr);

  if (shape_.IsTuple() || ShapeUtil::IsScalar(shape_)) {
    return LoopEmitter::EmitIndexAndSetExitBasicBlock(loop_name, index_type);
  }

  CHECK(!shape_.IsTuple());
  CHECK(!ShapeUtil::IsScalar(shape_));
  llvm_ir::ForLoopNest loop_nest(loop_name, b_);
  const int64 num_dims = shape_.dimensions_size();
  std::vector<llvm::Value*> array_multi_index(num_dims);

  // Add loops from outer-most to inner-most dimensions.
  bool outermost_loop = true;
  for (int i = LayoutUtil::MinorToMajor(shape_).size() - 1; i >= 0; --i) {
    const int64 dimension = LayoutUtil::Minor(shape_.layout(), i);
    const int bounds_index = num_dims - 1 - i;
    // Emit static loop bounds for this dimension.
    std::unique_ptr<llvm_ir::ForLoop> loop = loop_nest.AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape_.dimensions(dimension),
        /*suffix=*/absl::StrFormat("dim.%d", dimension),
	/*unroll_mode=*/xla::llvm_ir::UnrollMode::kDefaultUnroll,
	/*prevent_vectorization=*/false,
	/*tapir_loop=*/true, /*needs_sync=*/true);
    array_multi_index[dimension] = loop->GetIndVarValue();
    outermost_loop = false;
  }
  // Point IR builder at inner loop BB.
  llvm_ir::SetToFirstInsertPoint(loop_nest.GetInnerLoopBodyBasicBlock(), b_);

  // Set exit_bb_ to the exit block of the loop nest.
  exit_bb_ = loop_nest.GetOuterLoopExitBasicBlock();
  CHECK(exit_bb_ != nullptr);

  llvm_ir::IrArray::Index array_index(array_multi_index, shape_, index_type);
  return {array_index};
}

}  // namespace cpu
}  // namespace xla
