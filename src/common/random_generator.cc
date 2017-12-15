#include "./random_generator.h"

namespace mxnet {
namespace common {

template<typename DType>
void RandGeneratorSeed<cpu, DType>(RandGenerator<cpu, DType> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

template<typename DType>
RandGenerator<cpu, DType> *NewRandGenerator<cpu, DType>() {
  return new RandGenerator<cpu, DType>();
}

}  // namespace common
}  // namespace mxnet
