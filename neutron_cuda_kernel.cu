#include "neutron_cuda_kernel.h"

#include <cuda.h>
#include <cooperative_groups.h>

// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics

using namespace cooperative_groups;

__global__
void neutron_cuda_kernel(
		long n,
		int neutronsPerThread,
		const ProblemParameters* params,
		unsigned long long int* next_absorbed,
		float* absorbed,
		unsigned long long int* d_r,
		unsigned long long int* d_b,
		unsigned long long int* d_t,
		unsigned long long* seeds
) {
	const long id = blockIdx.x*blockDim.x + threadIdx.x;

	curandState state;
	curand_init(seeds[id], 0, 0, &state);

	const float c   = params->c;
	const float c_c = params->c_c;
	const float h   = params->h;

	unsigned int r = 0, b = 0, t = 0; // int is enough for local counts

	const long k = id*neutronsPerThread;
	const long m = min(static_cast<long>(neutronsPerThread), n-k);
	for (long i=0; i<m; i++) {

		float d = 0.0;
		float x = 0.0;

		float v;
		while (1) {

			const float u = curand_uniform (&state);
			const float L = -(1 / c) * log(u);
			x = x + L * cos(d);

			v = NO_VAL;
			if (x < 0) {
				r++;
				break;
			}
			else if (x >= h) {
				t++;
				break;
			}
			else if (curand_uniform (&state) < c_c / c) {
				b++;
				v = x;
				break;
			}
			else {
				const float u = curand_uniform (&state);
				d = u * M_PI;
			}
		}

		// save values to global memory
		if (v != NO_VAL) {
			auto g = coalesced_threads();
			unsigned long long int pos;
			if (g.thread_rank() == 0)
				pos = atomicAdd(next_absorbed, g.size());

			absorbed[g.shfl(pos, 0) + g.thread_rank()] = v;
		}
	}

	atomicAdd(d_r, static_cast<unsigned long long int>(r));
	atomicAdd(d_b, static_cast<unsigned long long int>(b));
	atomicAdd(d_t, static_cast<unsigned long long int>(t));
}
