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
 * \file sampler.h
 * \brief implementations of random sampling functors.
 */

#ifndef MXNET_OPERATOR_RANDOM_SAMPLER_H_
#define MXNET_OPERATOR_RANDOM_SAMPLER_H_

using namespace mshadow;
using namespace mxnet::op::mxnet_op;
using namespace mxnet::common::random;

namespace mxnet {
namespace op {

template<typename xpu>
struct SampleUniformKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, OType> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *lower, const IType *upper, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    out[i] = OType(lower[i/nBatch] + (upper[i/nBatch] - lower[i/nBatch]) * gen->uniform());
  }
};

template<typename xpu>
struct UniformSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lower,
                                   const Tensor<xpu, 1, IType>& upper,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    Kernel<SampleUniformKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, pgen, out.size(0), lower.size(0), out.size(0),
                                    lower.dptr_, upper.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleNormalKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, OType> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *mean, const IType *std, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    out[i] = OType(gen->normal() * std[i/nBatch] + mean[i/nBatch]);
  }
};

template<typename xpu>
struct NormalSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mean,
                                   const Tensor<xpu, 1, IType>& std,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    Kernel<SampleNormalKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, pgen, out.size(0), mean.size(0), out.size(0),
                                    mean.dptr_, std.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleExponentialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, OType> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *lambda, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    out[i] = OType(-log(1.0-gen->uniform()) / lambda[i/nBatch]);
  }
};

template<typename xpu>
struct ExponentialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    Kernel<SampleExponentialKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, pgen, out.size(0), lambda.size(0), out.size(0),
                                    lambda.dptr_, out.dptr_);
  }
};

template<typename xpu, typename IType, typename OType>
MSHADOW_XINLINE OType SampleGamma(IType a, IType b, RandGenerator<xpu, OType> *gen) {
  // Generate one sample of the gamma distribution
  OType sample;
  OType d = a < 1 ? a + 2.0 / 3.0 : a - 1.0 / 3.0;
  OType k = sqrt(9.0 * d);
  OType c = 1.0 / k;
  while (1) {
    OType Z = gen->normal();
    if (Z > -k) {
      OType x = 1.0 + c * Z;
      OType V = x * x * x;
      if (log(1.0-gen->uniform()) < 0.5 * Z * Z + d * (1.0 - V + log(V))) {
        sample = d * V * b;
        break;
      }
    }
  }
  return a < 1 ? sample * pow(gen->uniform(), OType(1.0 / a)) : sample;
}

template<typename xpu>
struct SampleGammaKernel {
  template<typename IType, typename OType, typename FType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, FType> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *alpha, const IType *beta, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    out[i] = OType(SampleGamma(alpha[i/nBatch], beta[i/nBatch], gen));
  }
};

template<typename xpu>
struct GammaSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, IType>& beta,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    typedef typename std::conditional<std::is_floating_point<OType>::value,
                                      OType, float>::type FType;
    RandGeneratorHost<xpu, FType> *gen = reinterpret_cast<RandGeneratorHost<xpu, FType> *>(pgen);
    Kernel<SampleGammaKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, gen, out.size(0), alpha.size(0), out.size(0),
                                    alpha.dptr_, beta.dptr_, out.dptr_);
  }
};

template<typename xpu>
MSHADOW_XINLINE int SamplePoisson(float lambda, RandGenerator<xpu, float> *gen) {
  // Generate one sample of the poisson distribution. Intentionally written
  // towards a specific type (float) for internal computation which is sufficient
  // for accurate enough computation.
  if ( lambda < 12.0 ) {
    float t = expf(-lambda);
    int x = 0;
    for ( float prod = gen->uniform(); prod > t; prod *= gen->uniform() ) { x += 1; }
    return x;
  } else {
    // Approximation for high lambda according to:
    // Numerical Recipes in C: The Art of Scientific Computing
    // Cambridge University Press
    const float pi(3.1415926);
    const float sq(sqrt(2.0*lambda));
    const float loglambda(log(lambda));
    const float g(lambda*loglambda-lgammaf(lambda+1.0));
    float em(0), t(0), y(0);
    do {
      do {
        y = tanf(pi * gen->uniform());
        em = sq * y + lambda;
      } while (em < 0.0);
      em = floorf(em);
      t = 0.9 * (1.0 + y * y) * expf(em * loglambda - lgammaf(em + 1.0) - g);
    } while (gen->uniform() > t);
    return static_cast<int>(em);
  }
}

template<typename xpu>
struct SamplePoissonKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, float> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *lambda, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    out[i] = OType(SamplePoisson(lambda[i/nBatch], gen));
  }
};

template<typename xpu>
struct PoissonSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& lambda,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGeneratorHost<xpu, float> *gen = reinterpret_cast<RandGeneratorHost<xpu, float> *>(pgen);
    Kernel<SamplePoissonKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, gen, out.size(0), lambda.size(0), out.size(0),
                                    lambda.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, float> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *k, const IType *p, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    float alpha = k[i/nBatch];
    float prob = p[i/nBatch];
    float beta = (1.0 - prob) / prob;
    float lambda = SampleGamma(alpha, beta, gen);
    out[i] = OType(SamplePoisson(lambda, gen));
  }
};

template<typename xpu>
struct NegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& k,
                                   const Tensor<xpu, 1, IType>& p,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGeneratorHost<xpu, float> *gen = reinterpret_cast<RandGeneratorHost<xpu, float> *>(pgen);
    Kernel<SampleNegativeBinomialKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, gen, out.size(0), k.size(0), out.size(0),
                                    k.dptr_, p.dptr_, out.dptr_);
  }
};

template<typename xpu>
struct SampleGeneralizedNegativeBinomialKernel {
  template<typename IType, typename OType>
  MSHADOW_XINLINE static void Map(int i, RandGenerator<xpu, float> *gen,
                                  index_t nParm, index_t nSample,
                                  const IType *mu, const IType *alpha, OType *out) {
    index_t nBatch(1 + (nSample - 1) / nParm);
    float lambda = alpha[i/nBatch] == 0 ? static_cast<float>(mu[i/nBatch])
            : SampleGamma(IType(1) / alpha[i/nBatch], alpha[i/nBatch] * mu[i/nBatch], gen);
    out[i] = OType(SamplePoisson(lambda, gen));
  }
};

template<typename xpu>
struct GeneralizedNegativeBinomialSampler {
  template<typename IType, typename OType>
  MSHADOW_FORCE_INLINE void Sample(const Tensor<xpu, 1, IType>& mu,
                                   const Tensor<xpu, 1, IType>& alpha,
                                   const Tensor<xpu, 1, OType>& out,
                                   RandGeneratorHost<xpu, OType> *pgen,
                                   Stream<xpu> *s) {
    RandGeneratorHost<xpu, float> *gen = reinterpret_cast<RandGeneratorHost<xpu, float> *>(pgen);
    Kernel<SampleGeneralizedNegativeBinomialKernel<xpu>, xpu>
      ::LaunchNativeRandomGenerator(s, gen, out.size(0), mu.size(0), out.size(0),
                                    mu.dptr_, alpha.dptr_, out.dptr_);
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RANDOM_SAMPLER_H_
