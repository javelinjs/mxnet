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
 * \file random_generator.cc
 * \brief cpu implements for random number generator.
 */

#include "./random_generator.h"

namespace mxnet {
namespace common {
namespace random {

template<typename DType>
class RandGenerator<cpu, DType> {
public:
  typedef typename std::conditional<std::is_floating_point<DType>::value,
  DType, float>::type FType;

  explicit RandGenerator() {}

  MSHADOW_XINLINE void Seed(uint32_t seed, uint32_t idx) { engine.seed(seed); }

  MSHADOW_XINLINE int rand() { return engine(); }

  MSHADOW_XINLINE FType uniform() { return uniformNum(engine); }

  MSHADOW_XINLINE FType normal() { return normalNum(engine); }

private:
  std::mt19937 engine;
  std::uniform_real_distribution<FType> uniformNum;
  std::normal_distribution<FType> normalNum;
};

template<>
void RandGeneratorSeed<cpu, float>(Stream<cpu> *s,
                                   RandGenerator<cpu, float> *gen,
                                   uint32_t seed) {
  gen->Seed(seed, 0);
}

template<>
RandGenerator<cpu, float> *NewRandGenerator<cpu, float>() {
  return new RandGenerator<cpu, float>();
}

template<>
void DeleteRandGenerator<cpu, float>(RandGenerator<cpu, float> *p) {
  if (p) delete p;
}

}  // namespace random
}  // namespace common
}  // namespace mxnet
