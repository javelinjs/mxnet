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
 * \brief gpu util functions for random number generator.
 */

#include <algorithm>
#include "./random_generator.h"
#include "../operator/mxnet_op.h"
#include "./cuda_utils.h"

namespace mxnet {
namespace common {
namespace random {

// always use with mxnet_op::LaunchNativeRandomGenerator
// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
public:
  __device__ __host__ explicit RandGenerator(curandStatePhilox4_32_10_t state)
  : state_(state) {}
  __device__ __host__ explicit RandGenerator() {}

  MSHADOW_FORCE_INLINE __device__ int rand() {
    return curand(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ float uniform() {
    return static_cast<float>(1.0) - curand_uniform(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ float normal() {
    return curand_normal(&state_);
  }

  MSHADOW_XINLINE __device__ curandStatePhilox4_32_10_t get_state() {
    return state_;
  }

private:
  curandStatePhilox4_32_10_t state_;
};

template<>
class RandGenerator<gpu, double> {
public:
  __device__ __host__ explicit RandGenerator(
  curandStatePhilox4_32_10_t state) : state_(state) {}
  __device__ __host__ explicit RandGenerator() {}

  MSHADOW_FORCE_INLINE __device__ int rand() {
    return curand(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ double uniform() {
    return static_cast<double>(1.0) - curand_uniform_double(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ double normal() {
    return curand_normal_double(&state_);
  }

  MSHADOW_XINLINE __device__ curandStatePhilox4_32_10_t get_state() {
    return state_;
  }

private:
  curandStatePhilox4_32_10_t state_;
};

template<typename DType>
class RandGeneratorGlobal<gpu, DType> : public RandGenerator<gpu, DType> {
public:
  __device__ __host__ explicit RandGeneratorGlobal() {}

  MSHADOW_FORCE_INLINE __device__ void Seed(uint32_t seed, uint32_t state_idx) {
    if (state_idx < kGPURndStateNum) curand_init(seed, state_idx, 0, &states_[state_idx]);
  }

  MSHADOW_FORCE_INLINE __device__ curandStatePhilox4_32_10_t get_state(uint32_t idx) {
    return states_[idx];
  }

  MSHADOW_FORCE_INLINE __device__ void set_state(curandStatePhilox4_32_10_t state,
                                                 uint32_t idx) {
    states_[idx] = state;
  }

private:
  // sizeof(curandStatePhilox4_32_10_t) = 64
  // sizeof(curandState_t) = 48
  // while for a large amount of states,
  // curand_init(curandState_t *) allocates extra memories on device.
  curandStatePhilox4_32_10_t states_[kGPURndStateNum];
};

template<>
class RandGeneratorGlobal<gpu, double> : public RandGenerator<gpu, double> {
public:
  __device__ __host__ explicit RandGeneratorGlobal() {}

  MSHADOW_FORCE_INLINE __device__ void Seed(uint32_t seed, uint32_t state_idx) {
    if (state_idx < kGPURndStateNum) curand_init(seed, state_idx, 0, &states_[state_idx]);
  }

  MSHADOW_FORCE_INLINE __device__ curandStatePhilox4_32_10_t get_state(uint32_t idx) {
    return states_[idx];
  }

  MSHADOW_FORCE_INLINE __device__ void set_state(curandStatePhilox4_32_10_t state,
                                                 uint32_t idx) {
    states_[idx] = state;
  }

private:
  curandStatePhilox4_32_10_t states_[kGPURndStateNum];
};

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