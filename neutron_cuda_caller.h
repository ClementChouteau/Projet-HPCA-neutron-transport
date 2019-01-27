#ifndef NEUTRON_CUDA_CALLER_H
#define NEUTRON_CUDA_CALLER_H

#include <vector>

ExperimentalResults neutron_cuda_caller
(
		float* absorbed,
		long n,
		const ProblemParameters& params,
		const std::vector<unsigned long long>& seeds,
		int threadsPerBlock,
		int neutronsPerThread
);

#endif // NEUTRON_CUDA_CALLER_H
