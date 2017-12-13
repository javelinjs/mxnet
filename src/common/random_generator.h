#ifndef MXNET_RANDOM_GENERATOR_H_
#define MXNET_RANDOM_GENERATOR_H_

#include <mxnet/base.h>
#include <random>

#ifdef __CUDACC__
#include "./cuda_utils.h"
#include <curand.h>
#include <curand_kernel.h>
#endif  // __CUDACC__

using namespace mshadow;

// Elementary random number generation for int/uniform/gaussian in CPU and GPU.
// Will use float data type whenever instantiated for half_t or any other non
// standard real type.
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator;

template<typename DType>
class RandGenerator<cpu, DType> {
public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
  DType, float>::type FType;
  std::mt19937 engine;
  std::uniform_real_distribution<FType> uniformNum;
  std::normal_distribution<FType> normalNum;
  explicit RandGenerator(unsigned int seed): engine(seed) {}
  MSHADOW_XINLINE void Seed(unsigned int seed) { engine.seed(seed); }
  MSHADOW_XINLINE int rand() { return engine(); }
  MSHADOW_XINLINE FType uniform() { return uniformNum(engine); }
  MSHADOW_XINLINE FType normal() { return normalNum(engine); }
};

#ifdef __CUDACC__

#define CURAND_STATE_SIZE 64

// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
public:
  __device__ RandGenerator(unsigned int seed) : seed_(seed) {}
  MSHADOW_FORCE_INLINE __device__ void init(unsigned int subsequence, unsigned int offset) {
    curand_init(seed_, subsequence, offset, &state_);
  }
  MSHADOW_FORCE_INLINE __device__ int rand(unsigned int i) { return curand(&state_); }
  MSHADOW_FORCE_INLINE __device__ float uniform(unsigned int i)
  { return static_cast<float>(1.0) - curand_uniform(&state_); }
  MSHADOW_FORCE_INLINE __device__ float normal(unsigned int i) { return curand_normal(&state_); }
private:
  curandState_t[CURAND_STATE_SIZE] states_;
};

template<>
class RandGenerator<gpu, double> {
public:
  __device__ RandGenerator(unsigned int seed) : seed_(seed) {}
  MSHADOW_FORCE_INLINE __device__ void init(unsigned int subsequence, unsigned int offset) {
    curand_init(seed_, subsequence, offset, &state_);
  }
  MSHADOW_FORCE_INLINE __device__ int rand() { return curand(&state_); }
  MSHADOW_FORCE_INLINE __device__ double uniform()
  { return static_cast<double>(1.0) - curand_uniform_double(&state_); }
  MSHADOW_FORCE_INLINE __device__ double normal() { return curand_normal_double(&state_); }
private:
  unsigned int seed_;
  curandState_t state_;
};

__global__ void RandGeneratorInit(RandGenerator<gpu> *pgen, unsigned int seed) {

}

#endif  // __CUDACC__

#endif  // MXNET_RANDOM_GENERATOR_H_
