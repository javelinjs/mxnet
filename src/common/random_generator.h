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

#ifdef __CUDACC__
#include <curand_kernel.h>
#endif  // __CUDACC__

using namespace mshadow;

namespace mxnet {
namespace common {
namespace random {

template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGeneratorHost;

// Elementary random number generation for int/uniform/gaussian in CPU and GPU.
// Will use float data type whenever instantiated for half_t or any other non
// standard real type.
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator;

// at least how many random numbers should be generated by one CPU thread.
const int kCPUMinRndNumberPerThread = 64;
// store how many global random states for CPU.
const int kCPURndStateNum = 1024;

template<typename DType>
class RandGenerator<cpu, DType> {
 public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
                                    DType, double>::type FType;

  explicit RandGenerator<cpu, DType>(std::mt19937 *ptr_engine) : engine_(ptr_engine) {}

  MSHADOW_XINLINE int rand() { return engine_->operator()(); }

  MSHADOW_XINLINE FType uniform() {
    typedef typename std::conditional<std::is_integral<DType>::value,
                                      std::uniform_int_distribution<DType>,
                                      std::uniform_real_distribution<FType>>::type GType;
    GType dist_uniform;
    return dist_uniform(*engine_);
  }

  MSHADOW_XINLINE FType normal() {
    std::normal_distribution<FType> dist_normal;
    return dist_normal(*engine_);
  }

 private:
  std::mt19937 *engine_;
};

template<typename DType>
class RandGeneratorHost<cpu, DType> {
public:
  RandGeneratorHost() {
    states_ = new std::mt19937[kCPURndStateNum];
  }

  ~RandGeneratorHost() {
    MSHADOW_CATCH_ERROR(delete[] states_);
  }

  MSHADOW_XINLINE RandGenerator<cpu, DType> Get(int idx = 0) {
    std::mt19937 *ptr_engine = states_ + idx;
    RandGenerator<cpu, DType> gen(ptr_engine);
    return gen;
  }

  MSHADOW_XINLINE void Seed(Stream<cpu> *, uint32_t seed) {
    for (int i = 0; i < kCPURndStateNum; ++i) (states_ + i)->seed(seed + i);
  }

  MSHADOW_XINLINE void set_state(int idx, std::mt19937 *state) {
    states_[idx] = *state;
  }

private:
  std::mt19937 *states_;
};

#ifdef __CUDACC__

// at least how many random numbers should be generated by one GPU thread.
const int kGPUMinRndNumberPerThread = 64;
// store how many global random states for GPU.
const int kGPURndStateNum = 32768;

// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
 public:
  // Copy state to local memory for efficiency.
  __device__ explicit RandGenerator(curandStatePhilox4_32_10_t *state)
      : state_(*state) {}

  MSHADOW_FORCE_INLINE __device__ int rand() {
    return curand(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ float uniform() {
    return static_cast<float>(1.0) - curand_uniform(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ float normal() {
    return curand_normal(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ curandStatePhilox4_32_10_t get_state() {
    return state_;
  }

 private:
  curandStatePhilox4_32_10_t state_;
};

template<>
class RandGenerator<gpu, double> {
 public:
  // Copy state to local memory for efficiency.
  __device__ explicit RandGenerator(curandStatePhilox4_32_10_t *state)
      : state_(*state) {}

  MSHADOW_FORCE_INLINE __device__ int rand() {
    return curand(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ double uniform() {
    return static_cast<double>(1.0) - curand_uniform_double(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ double normal() {
    return curand_normal_double(&state_);
  }

  MSHADOW_FORCE_INLINE __device__ curandStatePhilox4_32_10_t get_state() {
    return state_;
  }

 private:
  curandStatePhilox4_32_10_t state_;
};

template<typename DType>
class RandGeneratorHost<gpu, DType> {
 public:
  RandGeneratorHost();
  ~RandGeneratorHost();

  MSHADOW_FORCE_INLINE __device__ RandGenerator<gpu, DType> Get(int idx = 0) {
    curandStatePhilox4_32_10_t *ptr_state = states + idx;
    RandGenerator<gpu, DType> gen(ptr_state);
    return gen;
  }

  void Seed(Stream<gpu> *s, uint32_t seed);

  MSHADOW_FORCE_INLINE __device__ void set_state(int idx, curandStatePhilox4_32_10_t &state) {
    states_[idx] = state;
  }

 private:
  // sizeof(curandStatePhilox4_32_10_t) = 64
  // sizeof(curandState_t) = 48
  // while for a large amount of states, we notice
  // curand_init(curandState_t *) allocates extra memories on device,
  // (which is not mentioned in Nvidia's documents).
  // Thus we use curandStatePhilox4_32_10_t here to reduce GPU memory usage.
  curandStatePhilox4_32_10_t *states_;
  static __global__ void rand_generator_seed_kernel(curandStatePhilox4_32_10_t *, uint32_t);
};

#endif  // __CUDACC__

}  // namespace random
}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_RANDOM_GENERATOR_H_
