#ifndef NEUTRON_GPU_H
#define NEUTRON_GPU_H

#include <string>

#include "neutron.h"

ExperimentalResults neutron_gpu(float* absorbed, long n,
																const ProblemParameters& params,
																int threadsPerBlock, int neutronsPerThread,
																std::string oclDeviceType=std::string(""));

#endif // NEUTRON_GPU_H
