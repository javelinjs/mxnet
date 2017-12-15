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
 * \brief cpu functions for random number generator.
 */

#include "./random_generator.h"

namespace mxnet {
namespace common {
namespace random {

template<>
inline void RandGeneratorSeed<cpu, float>(RandGenerator<cpu, float> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

template<>
inline RandGenerator<cpu, float> *NewRandGenerator<cpu, float>() {
  return new RandGenerator<cpu, float>();
}

template<>
inline void DeleteRandGenerator(RandGenerator<cpu, float> *p) {
  if (p) delete p;
}

}  // namespace random
}  // namespace common
}  // namespace mxnet
