#ifndef NEUTRON_OPENCL_CALLER_H
#define NEUTRON_OPENCL_CALLER_H

#include <vector>
#include <string>

#include "neutron.h"

ExperimentalResults neutron_opencl_caller
(
		float* absorbed,
		long n,
		const ProblemParameters& params,
		const std::vector<unsigned long long>& seeds,
		int threadsPerBlock,
		int neutronsPerThread,
		std::string oclDeviceType
);

#endif // NEUTRON_OPENCL_CALLER_H
