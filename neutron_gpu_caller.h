#ifndef NEUTRON_GPU_CALLER_H
#define NEUTRON_GPU_CALLER_H

#include <vector>

ExperimentalResults neutron_gpu_caller(float* absorbed, int n,
																			 const ProblemParameters& params,
																			 const std::vector<unsigned long long>& seeds,
																			 int threadsPerBlock, int neutronsPerThread);

#endif // NEUTRON_GPU_CALLER_H
