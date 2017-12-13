/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file utils.cu
 * \brief gpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/tensor/cast_storage-inl.h"
#include "./random_generator.h"

template<typename DType>
__global__ void RandGeneratorInit(RandGenerator<gpu, DType> *pgen, unsigned int seed) {
  for (int i = 0; i < 64; ++i) {
    curand_init(seed, 0, 0, &(pgen->states_[i]));
  }
}

void RndInit(RandGenerator<gpu, float> *pgen, unsigned int global_seed) {
  RandGeneratorInit<<<1, 1>>>(pgen, global_seed);
}

void RndInit(RandGenerator<cpu, float> *pgen, unsigned int global_seed) {
}

void RndInit(RandGenerator<gpu, double> *pgen, unsigned int global_seed) {
  RandGeneratorInit<<<1, 1>>>(pgen, global_seed);
}

void RndInit(RandGenerator<cpu, double> *pgen, unsigned int global_seed) {
}

namespace mxnet {
namespace common {

template<>
void CheckFormatWrapper<gpu>(const RunContext &rctx, const NDArray &input,
                             const TBlob &err_cpu,  const bool full_check) {
  CheckFormatImpl<gpu>(rctx, input, err_cpu, full_check);
}

template<>
void CastStorageDispatch<gpu>(const OpContext& ctx,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl<gpu>(ctx, input, output);
}

}  // namespace common
}  // namespace mxnet
