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
 * \brief cpu util functions for random number generator.
 */

#include "./random_generator.h"

namespace mxnet {
namespace common {
namespace random {

template<typename DType>
void RandGeneratorSeed(RandGenerator<cpu, DType> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

template<typename DType>
RandGenerator<cpu, DType> *NewRandGenerator() {
  return new RandGenerator<cpu, DType>();
}

template<typename DType>
void DeleteRandGenerator(RandGenerator<cpu, DType> *p) {
  if (p) delete p;
}

}  // namespace random
}  // namespace common
}  // namespace mxnet
