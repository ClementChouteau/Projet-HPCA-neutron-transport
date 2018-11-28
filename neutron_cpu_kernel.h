#ifndef NEUTRON_CPU_KERNEL_H
#define NEUTRON_CPU_KERNEL_H

#include "neutron.h"

#include <random>
#include <cmath>

template<typename RandomGenerator>
ExperimentalResults neutron_cpu_kernel(int n, const ProblemParameters& params,
																			 float* absorbed, int size,
																			 RandomGenerator& rng) {
	ExperimentalResults res;
	res.absorbed = absorbed;
	res.r = res.b = res.t = 0;

	auto uniform_0_1 = std::uniform_real_distribution<float>(0., 1.);

	int cpt = 0;
	for (int i=0; i<n && cpt<size; i++) {

		float d = 0.0;
		float x = 0.0;

		while (1) {

			const float u = uniform_0_1(rng);
			float L = -(1 / params.c) * std::log(u);
			x = x + L * std::cos(d);
			if (x < 0) {
				res.r++;
				break;
			}
			else if (x >= params.h) {
				res.t++;
				break;
			}
			else if (uniform_0_1(rng) < (params.c_c / params.c)) {
				res.b++;
				res.absorbed[cpt++] = x;
				break;
			}
			else {
				const float u = uniform_0_1(rng);
				d = u * M_PI;
			}
		}
	}

	return res;
}

#endif // NEUTRON_CPU_KERNEL_H
