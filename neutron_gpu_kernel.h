#ifndef NEUTRON_GPU_KERNEL_H
#define NEUTRON_GPU_KERNEL_H

#include "neutron.h"

#include <vector>

#include <curand_kernel.h>

__global__
void neutron_seq_kernel(long n,
						int neutronsPerThread,
						const ProblemParameters* params,
						float* absorbed,
						long* d_r,
						long* d_b,
						long* d_t,
						unsigned long long* seeds,
						curandState* states);

#endif // NEUTRON_GPU_KERNEL_H
