#include "neutron_gpu_kernel.h"

#include <cuda.h>

__global__
void neutron_seq_kernel(long n,
												int neutronsPerThread,
												const ProblemParameters* params,
												float* absorbed,
												long* d_r,
												long* d_b,
												long* d_t,
												unsigned long long* seeds,
												curandState* states) {
	const long id = blockIdx.x*blockDim.x + threadIdx.x;

	curand_init(seeds[id], id, 0, states+id);

	const float c = params->c;
	const float c_c = params->c_c;
	const float h = params->h;
	long r = 0, b = 0, t = 0;

	long cpt = (blockIdx.x*blockDim.x)*neutronsPerThread + threadIdx.x;
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
		absorbed[cpt+=blockDim.x] = v;
	}

	d_r[id] = r;
	d_b[id] = b;
	d_t[id] = t;
}
