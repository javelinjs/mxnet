#include "./random_generator.h"

namespace mxnet {
namespace common {

template<>
void RandGeneratorSeed<cpu, float>(RandGenerator<cpu, float> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

template<>
RandGenerator<cpu, float> *NewRandGenerator<cpu, float>() {
  return new RandGenerator<cpu, float>();
}

template<>
void RandGeneratorSeed<cpu, double>(RandGenerator<cpu, double> *gen, unsigned int seed) {
  gen->Seed(seed, 0);
}

template<>
RandGenerator<cpu, double> *NewRandGenerator<cpu, double>() {
  return new RandGenerator<cpu, double>();
}

}  // namespace common
}  // namespace mxnet
