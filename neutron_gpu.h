#ifndef NEUTRON_GPU_H
#define NEUTRON_GPU_H

#include "neutron.h"

ExperimentalResults neutron_gpu(float* absorbed, long n,
																const ProblemParameters& params,
																int threadsPerBlock, int neutronsPerThread);

#endif // NEUTRON_GPU_H
