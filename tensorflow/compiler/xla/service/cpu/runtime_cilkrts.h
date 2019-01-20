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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CILKRTS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CILKRTS_H_

#include <cstdint>
#include "tensorflow/core/platform/types.h"

// TODO: Include a relevant Cilk header file?
struct __cilkrts_pedigree {
  uint64_t rank;
  __cilkrts_pedigree *next;
};

struct __cilkrts_stack_frame;
struct __cilkrts_worker {
  __cilkrts_stack_frame **tail;
  __cilkrts_stack_frame **head;
  __cilkrts_stack_frame **exc;
  __cilkrts_stack_frame **protected_tail;
  __cilkrts_stack_frame **ltq_limit;
  int32_t self;
  void *g;
  void *l;
  void *reducer_map;
  __cilkrts_stack_frame *current_stack_frame;
  void *saved_protected_tail;
  void *sysdep;
  __cilkrts_pedigree pedigree;
};

struct __cilkrts_stack_frame {
  uint32_t flags;
  int32_t size;
  __cilkrts_stack_frame *call_parent;
  __cilkrts_worker *worker;
  void *except_data;
  void *ctx[5];
  uint32_t mxcsr;
  uint16_t fpcsr;
  uint16_t reserved;
  __cilkrts_pedigree parent_pedigree;
};

extern "C" __cilkrts_worker *__cilkrts_bind_thread_1();
extern "C" int __cilkrts_get_nworkers();
extern "C" __cilkrts_worker *__cilkrts_get_tls_worker();
extern "C" __cilkrts_worker *__cilkrts_get_tls_worker_fast();
extern "C" void __cilkrts_leave_frame(__cilkrts_stack_frame *);
extern "C" void __cilkrts_sync(__cilkrts_stack_frame *);

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CILKRTS_H_
