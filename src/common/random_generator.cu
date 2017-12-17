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
 * Copyright (c) 2017 by Contributors
 * \file random_generator.cu
 * \brief gpu implements for random number generator.
 */

#include <algorithm>
#include "./random_generator.h"
#include "../operator/mxnet_op.h"
#include "./cuda_utils.h"

namespace mxnet {
namespace common {
namespace random {

template<typename DType>
__global__ void rand_generator_seed_kernel(RandGeneratorGlobal<gpu, DType> *gen,
                                           uint32_t seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  gen->Seed(seed, id);
}

/*!
 * \brief Initialize states for RandGeneratorGlobal.
 * \tparam s gpu stream
 * \tparam gen pointer to RandGeneratorGlobal on device.
 * \tparam seed seed for curand
 */
template<>
void RandGeneratorSeed<gpu, float>(Stream<gpu> *s,
                                   RandGenerator<gpu, float> *gen,
                                   uint32_t seed) {
  using namespace mshadow::cuda;
  int ngrid = std::min(kMaxGridNum, (kGPURndStateNum + kBaseThreadNum - 1) / kBaseThreadNum);
  rand_generator_seed_kernel<<<ngrid, kBaseThreadNum, 0, Stream<gpu>::GetStream(s)>>>(
      reinterpret_cast<RandGeneratorGlobal<gpu, float> *>(gen), seed);
}

// allocate RandGeneratorGlobal on device
template<>
RandGenerator<gpu, float> *NewRandGenerator<gpu, float>() {
  RandGeneratorGlobal<gpu, float> *gen;
  CUDA_CALL(cudaMalloc(&gen, sizeof(RandGeneratorGlobal<gpu, float>)));
  return gen;
};

template<>
void DeleteRandGenerator<gpu, float>(RandGenerator<gpu, float> *p) {
  if (p) cudaFree(reinterpret_cast<RandGeneratorGlobal<gpu, float> *>(p));
}

}  // namespace random
}  // namespace common
}  // namespace mxnet