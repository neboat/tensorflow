/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"

#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace llvm_ir {

ForLoop::ForLoop(absl::string_view prefix, absl::string_view suffix,
                 llvm::Value* start_index, llvm::Value* end_index,
                 llvm::Value* step, UnrollMode unroll_mode,
                 bool prevent_vectorization, bool tapir_loop,
                 llvm::Value* sync_reg, bool needs_sync)
    : prefix_(prefix),
      suffix_(suffix),
      start_index_(start_index),
      end_index_(end_index),
      step_(step),
      insert_before_bb_(nullptr),
      unroll_mode_(unroll_mode),
      prevent_vectorization_(prevent_vectorization),
      tapir_loop_(tapir_loop),
      sync_reg_(sync_reg),
      needs_sync_(needs_sync) {}

/* static */ std::unique_ptr<ForLoop> ForLoop::EmitForLoop(
    absl::string_view prefix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* step, llvm::IRBuilder<>* b, UnrollMode unroll_mode,
    bool prevent_vectorization, bool tapir_loop, llvm::Value *sync_reg,
    bool needs_sync) {
  std::unique_ptr<ForLoop> loop(new ForLoop(prefix, /*suffix=*/"", start_index,
                                            end_index, step, unroll_mode,
                                            prevent_vectorization, tapir_loop,
                                            sync_reg, needs_sync));
  loop->Emit(b);
  return loop;
}

void ForLoop::Emit(llvm::IRBuilder<>* b) {
  // The preheader block is the block the builder is currently emitting
  // code into.
  preheader_bb_ = b->GetInsertBlock();

  llvm::BasicBlock::iterator insert_point = b->GetInsertPoint();
  if (insert_point == preheader_bb_->end()) {
    // We're emitting the loop at the end of a basic block. Verify there is no
    // terminator (eg, branch) in the basic block.
    CHECK_EQ(nullptr, preheader_bb_->getTerminator());

    exit_bb_ = CreateLoopBB("loop_exit", b);
  } else {
    // We're emitting the loop into the middle of a basic block. splitBasicBlock
    // requires that this basic block be well-formed (have a terminator).
    CHECK_NE(nullptr, preheader_bb_->getTerminator());

    // Split the preheader to create an exit basic block. The exit basic block
    // will contain all instructions at or after insert_point.
    exit_bb_ = preheader_bb_->splitBasicBlock(insert_point,
                                              GetQualifiedName("loop_exit"));

    // splitBasicBlock adds an unconditional branch between the split basic
    // blocks. Remove it. An unconditional branch will be added below from the
    // preheader to the header.
    preheader_bb_->getTerminator()->eraseFromParent();
  }
  insert_before_bb_ = exit_bb_;

  // Create remaining basic block which form the inside of the loop.
  header_bb_ = CreateLoopBB("loop_header", b);
  llvm::BasicBlock* detach_bb_ = nullptr;
  if (tapir_loop_) {
    detach_bb_ = CreateLoopBB("loop_detach", b);
  }
  body_bb_ = CreateLoopBB("loop_body", b);
  if (!tapir_loop_) {
    detach_bb_ = body_bb_;
  }
  llvm::BasicBlock* inc_bb_ = body_bb_;
  if (tapir_loop_) {
    inc_bb_ = CreateLoopBB("loop_inc", b);
  }
  llvm::BasicBlock* sync_bb_ = exit_bb_;
  if (tapir_loop_ && needs_sync_) {
    sync_bb_ = CreateLoopBB("loop_sync", b);
  }

  // Function entry basic block.
  // Emit alloca for the induction variable. We do this at the entry to the
  // basic block to ensure the alloc only executes once per function (we could
  // be emitting a nested loop).
  llvm::BasicBlock* task_entry = llvm::GetDetachedCtx(preheader_bb_);
  b->SetInsertPoint(task_entry,
                    task_entry->getFirstInsertionPt());
  llvm::Value* indvar_address = b->CreateAlloca(
      start_index_->getType(), nullptr, GetQualifiedName("indvar_address"));

  // Preheader basic block.
  // Initialize induction variable starting index. Create branch to the header.
  b->SetInsertPoint(preheader_bb_);
  b->CreateStore(start_index_, indvar_address);
  // The preheader should not have a branch yet.
  CHECK_EQ(preheader_bb_->getTerminator(), nullptr);
  b->CreateBr(header_bb_);

  // Header basic block.
  // Emit the loop conditional branch. Load and compare indvar with ending
  // index and jump to loop exit if equal. Jump to body otherwise.
  b->SetInsertPoint(header_bb_);
  indvar_ = b->CreateLoad(indvar_address, GetQualifiedName("indvar"));
  llvm::Value* exit_cond = b->CreateICmpUGE(indvar_, end_index_);
  b->CreateCondBr(/*Cond=*/exit_cond,
                  /*True=*/(tapir_loop_ && needs_sync_) ? sync_bb_ : exit_bb_,
                  /*False=*/tapir_loop_ ? detach_bb_ : body_bb_);

  // FIXME: Check for Tapir-loop emission properly.
  if (tapir_loop_) {
    // Detach basic block.
    // Detach the loop body.
    CHECK_NE(nullptr, detach_bb_);
    CHECK_NE(nullptr, sync_reg_);
    b->SetInsertPoint(detach_bb_);
    b->CreateDetach(body_bb_, inc_bb_, sync_reg_);

    // Body basic block.
    // Reattach the body to the increment basic block.
    CHECK_NE(body_bb_, inc_bb_);
    CHECK_NE(nullptr, sync_reg_);
    b->SetInsertPoint(body_bb_);
    b->CreateReattach(inc_bb_, sync_reg_);
  }

  // Increment basic block.
  // Increment indvar, store indvar, and jump to header.
  b->SetInsertPoint(inc_bb_);
  llvm::Value* step = step_;
  llvm::Value* indvar = indvar_;

  llvm::Value* indvar_inc = b->CreateAdd(indvar, step, "indvar.inc",
                                         /*HasNUW=*/true, /*HasNSW=*/true);
  b->CreateStore(indvar_inc, indvar_address);
  llvm::BranchInst* back_branch = b->CreateBr(header_bb_);

  std::vector<llvm::Metadata*> loop_metadata = GetLoopMetadata(b);
  if (!loop_metadata.empty()) {
    llvm::LLVMContext* ctx = &start_index_->getContext();
    auto temp_node = llvm::MDNode::getTemporary(*ctx, llvm::None);
    loop_metadata.insert(loop_metadata.begin(), temp_node.get());
    auto loop_id = llvm::MDNode::get(*ctx, loop_metadata);
    loop_id->replaceOperandWith(0, loop_id);
    back_branch->setMetadata(llvm::LLVMContext::MD_loop, loop_id);
  }

  // Add a sync instruction for the loop, if need be.
  if (tapir_loop_ && needs_sync_) {
    CHECK_NE(sync_bb_, exit_bb_);
    CHECK_NE(nullptr, sync_reg_);
    b->SetInsertPoint(sync_bb_);
    b->CreateSync(exit_bb_, sync_reg_);
  }

  // Re-point the IR builder to the loop exit block.
  b->SetInsertPoint(exit_bb_);
}

std::vector<llvm::Metadata*> ForLoop::GetLoopMetadata(llvm::IRBuilder<>* b) {
  const char* const kLlvmLoopUnrollDisableMDName = "llvm.loop.unroll.disable";
  const char* const kLlvmLoopUnrollFullMDName = "llvm.loop.unroll.full";
  const char* const kLlvmLoopVectorizeMDName = "llvm.loop.vectorize.enable";
  const char* const kLlvmTapirLoopSpawnStrategyMDName =
    "tapir.loop.spawn.strategy";
  // Use DAC spawning strategy for Tapir loops.
  // TODO: Generalize this code.
  int32 TapirDACLoopSpawning = 1;
  llvm::LLVMContext* ctx = &start_index_->getContext();

  std::vector<llvm::Metadata*> result;
  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kNoUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollDisableMDName)}));
  }

  if (prevent_vectorization_) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopVectorizeMDName),
               llvm::ConstantAsMetadata::get(b->getFalse())}));
  }

  if (unroll_mode_ == xla::llvm_ir::UnrollMode::kFullyUnroll) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmLoopUnrollFullMDName)}));
  }

  if (tapir_loop_) {
    result.push_back(llvm::MDNode::get(
        *ctx, {llvm::MDString::get(*ctx, kLlvmTapirLoopSpawnStrategyMDName),
               llvm::ConstantAsMetadata::get(
                   b->getInt32(TapirDACLoopSpawning))}));
  }
  return result;
}

string ForLoop::GetQualifiedName(absl::string_view name) {
  return llvm_ir::IrName(prefix_, llvm_ir::IrName(name, suffix_));
}

llvm::BasicBlock* ForLoop::CreateLoopBB(absl::string_view name,
                                        llvm::IRBuilder<>* b) {
  return CreateBasicBlock(insert_before_bb_, GetQualifiedName(name), b);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(absl::string_view suffix,
                                              llvm::Value* start_index,
                                              llvm::Value* end_index,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization,
                                              bool tapir_loop,
                                              bool needs_sync) {
  return AddLoop(suffix, start_index, end_index, GetConstantWithIndexType(1),
                 unroll_mode, prevent_vectorization, tapir_loop, needs_sync);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(
    absl::string_view suffix, llvm::Value* start_index, llvm::Value* end_index,
    llvm::Value* stride, UnrollMode unroll_mode, bool prevent_vectorization,
    bool tapir_loop, bool needs_sync) {
  llvm::Value* sync_reg = nullptr;
  if (inner_loop_body_bb_ != nullptr) {
    // Create this loop inside the previous one.
    b_->SetInsertPoint(&*inner_loop_body_bb_->getFirstInsertionPt());
    if (tapir_loop) {
      sync_reg =
        llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::syncregion_start, {}, {},
                                     b_);
    }
  } else if (tapir_loop) {
    // Function entry basic block.
    // Emit a sync region for the outermost loop.
    llvm::Function* func = b_->GetInsertBlock()->getParent();
    if (b_->GetInsertBlock() == &func->getEntryBlock()) {
      sync_reg =
        llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::syncregion_start, {}, {},
                                     b_);
    } else {
      // auto savedIP = b_->saveIP();
      // b_->SetInsertPoint(&func->getEntryBlock(),
      //                    func->getEntryBlock().getFirstInsertionPt());
      sync_reg =
        llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::syncregion_start, {}, {},
                                     b_);
      // b_->restoreIP(savedIP);
    }
  }
  std::unique_ptr<ForLoop> loop(new ForLoop(
      /*prefix=*/name_, suffix, start_index, end_index, stride, unroll_mode,
      prevent_vectorization, tapir_loop, sync_reg, needs_sync));
  loop->Emit(b_);

  if (outer_loop_preheader_bb_ == nullptr) {
    outer_loop_preheader_bb_ = loop->GetPreheaderBasicBlock();
  }

  if (outer_loop_exit_bb_ == nullptr) {
    outer_loop_exit_bb_ = loop->GetExitBasicBlock();
  }

  inner_loop_body_bb_ = loop->GetBodyBasicBlock();

  return loop;
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64 start_index,
                                              int64 end_index,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization,
                                              bool tapir_loop,
                                              bool needs_sync) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index), unroll_mode,
                 prevent_vectorization, tapir_loop, needs_sync);
}

std::unique_ptr<ForLoop> ForLoopNest::AddLoop(int64 start_index,
                                              int64 end_index, int64 stride,
                                              absl::string_view suffix,
                                              UnrollMode unroll_mode,
                                              bool prevent_vectorization,
                                              bool tapir_loop,
                                              bool needs_sync) {
  CHECK_LE(start_index, end_index);
  return AddLoop(suffix, GetConstantWithIndexType(start_index),
                 GetConstantWithIndexType(end_index),
                 GetConstantWithIndexType(stride), unroll_mode,
                 prevent_vectorization, tapir_loop, needs_sync);
}

IrArray::Index ForLoopNest::AddLoopsForShape(const Shape& shape,
                                             absl::string_view suffix,
                                             bool tapir_loop) {
  std::vector<int64> dimensions(shape.rank());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return IrArray::Index(AddLoopsForShapeOnDimensions(shape, dimensions, suffix,
                                                     tapir_loop),
                        shape, index_type_);
}

std::vector<llvm::Value*> ForLoopNest::AddLoopsForShapeOnDimensions(
    const Shape& shape, absl::Span<const int64> dimensions,
    absl::string_view suffix, bool tapir_loop) {
  std::vector<llvm::Value*> multi_index(shape.dimensions_size());
  for (int64 dimension : dimensions) {
    std::unique_ptr<llvm_ir::ForLoop> loop = AddLoop(
        /*start_index=*/0,
        /*end_index=*/shape.dimensions(dimension),
        /*suffix=*/
        llvm_ir::IrName(suffix, absl::StrCat(dimension)),
        /*unroll_mode=*/ llvm_ir::UnrollMode::kDefaultUnroll,
        /*prevent_vectorization=*/ false,
        /*tapir_loop=*/ tapir_loop, /*needs_sync=*/ true);
    multi_index[dimension] = loop->GetIndVarValue();
  }
  return multi_index;
}

std::vector<llvm::Value*> ForLoopNest::EmitOperandArrayLoopNest(
    const llvm_ir::IrArray& operand_array, int64 dimension_to_skip,
    absl::string_view name_suffix, bool tapir_loop) {
  // Prepares the dimension list we will use to emit the loop nest. Outermost
  // loops are added first. Add loops in major-to-minor order, and skip the
  // 'dimension_to_skip' dimension.
  std::vector<int64> dimensions;
  const Shape& shape = operand_array.GetShape();
  // Initially get the dimensions in minor to major order, then reverse them.
  for (int64 dimension : LayoutUtil::MinorToMajor(shape)) {
    if (dimension != dimension_to_skip) {
      dimensions.push_back(dimension);
    }
  }
  absl::c_reverse(dimensions);

  // Create loop nest with one for-loop for each dimension of the
  // output.
  std::vector<llvm::Value*> multi_index =
      AddLoopsForShapeOnDimensions(shape, dimensions, name_suffix, tapir_loop);
  // Verify every dimension except the 'dimension_to_skip' dimension was set in
  // the index.
  for (size_t dimension = 0; dimension < multi_index.size(); ++dimension) {
    if (dimension == dimension_to_skip) {
      DCHECK_EQ(nullptr, multi_index[dimension]);
    } else {
      DCHECK_NE(nullptr, multi_index[dimension]);
    }
  }
  return multi_index;
}

}  // namespace llvm_ir
}  // namespace xla
