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
 * \brief gpu functions for random number generator.
 */

#include <algorithm>
#include <mshadow/cuda/tensor_gpu-inl.cuh>
#include "./random_generator.h"
#include "./cuda_utils.h"

namespace mxnet {
namespace common {

__global__ void rand_generator_seed_kernel(RandGenerator<gpu> *pgen, unsigned int seed) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  pgen->Seed(seed, id);
}

template<>
inline void RandGeneratorSeed(RandGenerator<gpu> *gen, unsigned int seed) {
  using namespace mshadow::cuda;
  int ngrid = std::min(kMaxGridNum, (CURAND_STATE_SIZE + kBaseThreadNum - 1) / kBaseThreadNum);
  rand_generator_seed_kernel<<<ngrid, kBaseThreadNum, 0, 0>>>(gen, seed);
}

template<>
inline RandGenerator<gpu> *NewRandGenerator() {
  RandGenerator<gpu> *gen;
  CUDA_CALL(cudaMalloc(&gen, sizeof(RandGenerator<gpu>)));
  return gen;
};

}  // namespace common
}  // namespace mxnet