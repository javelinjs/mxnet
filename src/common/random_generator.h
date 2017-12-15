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
 * \file random_generator.h
 * \brief Native random number generator.
 */
#ifndef MXNET_COMMON_RANDOM_GENERATOR_H_
#define MXNET_COMMON_RANDOM_GENERATOR_H_

#include <mxnet/base.h>
#include <random>

#if MXNET_USE_CUDA
#include <curand_kernel.h>
#endif  // MXNET_USE_CUDA

using namespace mshadow;

namespace mxnet {
namespace common {

// Elementary random number generation for int/uniform/gaussian in CPU and GPU.
// Will use float data type whenever instantiated for half_t or any other non
// standard real type.
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator;

template<typename xpu>
inline void RandGeneratorSeed(RandGenerator<xpu> *, unsigned int seed);

template<typename DType>
class RandGenerator<cpu, DType> {
public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
                                    DType, float>::type FType;
  explicit RandGenerator() {}
  MSHADOW_XINLINE void Seed(unsigned int seed, unsigned int idx) { engine.seed(seed); }
  MSHADOW_XINLINE int rand(unsigned int i = 0) { return engine(); }
  MSHADOW_XINLINE FType uniform(unsigned int i = 0) { return uniformNum(engine); }
  MSHADOW_XINLINE FType normal(unsigned int i = 0) { return normalNum(engine); }
private:
  std::mt19937 engine;
  std::uniform_real_distribution<FType> uniformNum;
  std::normal_distribution<FType> normalNum;
};

#ifdef MXNET_USE_CUDA
#define CURAND_STATE_SIZE 64

// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
public:
  __device__ __host__ explicit RandGenerator() {}
  MSHADOW_FORCE_INLINE __device__ void Seed(unsigned int seed, unsigned int state_idx) {
    if (state_idx < CURAND_STATE_SIZE) curand_init(seed, state_idx, 0, &(states_[state_idx]));
  }
  MSHADOW_FORCE_INLINE __device__ int rand(unsigned int i = 0) {
    return curand(&(states_[i % CURAND_STATE_SIZE]));
  }
  MSHADOW_FORCE_INLINE __device__ float uniform(unsigned int i = 0) {
    return static_cast<float>(1.0) - curand_uniform(&(states_[i % CURAND_STATE_SIZE]));
  }
  MSHADOW_FORCE_INLINE __device__ float normal(unsigned int i = 0) {
    return curand_normal(&(states_[i % CURAND_STATE_SIZE]));
  }
private:
  curandState_t states_[CURAND_STATE_SIZE];
};

template<>
class RandGenerator<gpu, double> {
public:
  __device__ __host__ explicit RandGenerator() {}
  MSHADOW_FORCE_INLINE __device__ void Seed(unsigned int seed, unsigned int state_idx) {
    if (state_idx < CURAND_STATE_SIZE) curand_init(seed, state_idx, 0, &(states_[state_idx]));
  }
  MSHADOW_FORCE_INLINE __device__ int rand(unsigned int i = 0) {
    return curand(&(states_[i % CURAND_STATE_SIZE]));
  }
  MSHADOW_FORCE_INLINE __device__ double uniform(unsigned int i = 0) {
    return static_cast<double>(1.0) - curand_uniform_double(&(states_[i % CURAND_STATE_SIZE]));
  }
  MSHADOW_FORCE_INLINE __device__ double normal(unsigned int i = 0) {
    return curand_normal_double(&(states_[i % CURAND_STATE_SIZE]));
  }
  curandState_t states_[CURAND_STATE_SIZE];
};
#endif  // MXNET_USE_CUDA

template<>
inline void RandGeneratorSeed(RandGenerator<cpu> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_RANDOM_GENERATOR_H_
