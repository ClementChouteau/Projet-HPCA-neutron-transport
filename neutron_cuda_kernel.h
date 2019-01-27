#ifndef NEUTRON_CUD_KERNEL_H
#define NEUTRON_CUD_KERNEL_H

#include "neutron.h"

#include <vector>

#include <curand_kernel.h>

__global__
void neutron_cuda_kernel
(
		long n,
		int neutronsPerThread,
		const ProblemParameters* params,
		unsigned long long int* d_next_absorbed,
		float* d_absorbed,
		unsigned long long int* d_r,
		unsigned long long int* d_b,
		unsigned long long int* d_t,
		unsigned long long* d_seeds
);

#endif // NEUTRON_CUD_KERNEL_H
