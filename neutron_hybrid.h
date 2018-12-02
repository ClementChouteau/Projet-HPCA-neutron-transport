#ifndef NEUTRON_HYBRID_H
#define NEUTRON_HYBRID_H

#include "neutron.h"

// ratio is a real number in [0, 1] indicating
// the proportion of cpu/gpu values to compute
ExperimentalResults neutron_hybrid(float* absorbed, long n,
																	 const ProblemParameters& params,
																	 int threadsPerBlock, int neutronsPerThread,
																	 float ratio);

#endif // NEUTRON_HYBRID_H
