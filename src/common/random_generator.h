#ifndef MXNET_RANDOM_GENERATOR_H_
#define MXNET_RANDOM_GENERATOR_H_

#include <mxnet/base.h>
#include <random>

//#ifdef __CUDACC__
#include "./cuda_utils.h"
#include <curand.h>
#include <curand_kernel.h>
//#endif  // __CUDACC__

using namespace mshadow;

//namespace mxnet {
//namespace common {

// Elementary random number generation for int/uniform/gaussian in CPU and GPU.
// Will use float data type whenever instantiated for half_t or any other non
// standard real type.
template<typename Device, typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator;

template<typename DType MSHADOW_DEFAULT_DTYPE>
class RandGenerator<cpu, DType> {
public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
  DType, float>::type FType;
  std::mt19937 engine;
  std::uniform_real_distribution<FType> uniformNum;
  std::normal_distribution<FType> normalNum;
  explicit RandGenerator(unsigned int seed): engine(seed) {}
  MSHADOW_XINLINE void Seed(unsigned int seed) { engine.seed(seed); }
  MSHADOW_XINLINE int rand(unsigned int i = 0) { return engine(); }
  MSHADOW_XINLINE FType uniform(unsigned int i = 0) { return uniformNum(engine); }
  MSHADOW_XINLINE FType normal(unsigned int i = 0) { return normalNum(engine); }
};


#define CURAND_STATE_SIZE 64

// uniform number generation in Cuda made consistent with stl (include 0 but exclude 1)
// by using 1.0-curand_uniform(). Needed as some samplers below won't be able to deal with
// one of the boundary cases.
template<typename DType>
class RandGenerator<gpu, DType> {
public:
  __device__ __host__ RandGenerator(unsigned int seed) {}
  MSHADOW_FORCE_INLINE __device__ void init(unsigned int subsequence, unsigned int offset) {
    // curand_init(seed_, subsequence, offset, states_);
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
  curandState_t states_[CURAND_STATE_SIZE];
};

template<>
class RandGenerator<gpu, double> {
public:
  __device__ __host__ RandGenerator() {}
  MSHADOW_FORCE_INLINE __device__ void init(unsigned int subsequence, unsigned int offset) {
    //curand_init(seed_, subsequence, offset, states_);
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
private:
  curandState_t states_[CURAND_STATE_SIZE];
};

//#endif  // __CUDACC__
//}
//}

#endif  // MXNET_RANDOM_GENERATOR_H_
