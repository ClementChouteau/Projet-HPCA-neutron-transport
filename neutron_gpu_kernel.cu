#include "neutron_gpu_kernel.h"

#include <cuda.h>
#include <cooperative_groups.h>

// The global atomics are supposed to be optimized by compiler (>= CUDA 9)
// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics

using namespace cooperative_groups;

__global__
void neutron_gpu_kernel(long n,
												int neutronsPerThread,
												const ProblemParameters* params,
												unsigned long long int* next_absorbed,
												float* absorbed,
												unsigned long long int* d_r,
												unsigned long long int* d_b,
												unsigned long long int* d_t,
												unsigned long long* seeds,
												curandState* states) {
	const long id = blockIdx.x*blockDim.x + threadIdx.x;

	curand_init(seeds[id], id, 0, states+id);

	const float c = params->c;
	const float c_c = params->c_c;
	const float h = params->h;
	unsigned long long int r = 0, b = 0, t = 0;

	long cpt = (blockIdx.x*blockDim.x + threadIdx.x)*neutronsPerThread;
	auto g = coalesced_threads();
	for (long i=0; i<neutronsPerThread; i++) {
		if (!(cpt < n))
			break;

		float d = 0.0;
		float x = 0.0;

		float v;
		while (1) {

			const float u = curand_uniform (states+id);
			float L = -(1 / c) * log(u);
			x = x + L * cos(d);

			if (x < 0) {
				r++;
				v = NO_VAL;
				break;
			}
			else if (x >= h) {
				t++;
				v = NO_VAL;
				break;
			}
			else if (curand_uniform (states+id) < c_c / c) {
				b++;
				v = x;
				break;
			}
			else {
				const float u = curand_uniform (states+id);
				d = u * M_PI;
			}
		}
		unsigned long long int pos;
		if(g.thread_rank() == 0)
			pos = atomicAdd(next_absorbed, g.size());

		if (v != NO_VAL) {
			absorbed[g.shfl(pos, 0) + g.thread_rank()] = v;
		}
		cpt++;
	}

	atomicAdd(d_r, r);
	atomicAdd(d_b, b);
	atomicAdd(d_t, t);
}
