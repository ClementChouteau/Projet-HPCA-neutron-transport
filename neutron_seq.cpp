#include "neutron_seq.h"

#include "neutron_cpu_kernel.h"

#include <random>

ExperimentalResults neutron_seq(float* absorbed, int n,
																const ProblemParameters& params) {
	std::mt19937 rng;
	rng.seed(std::random_device()());

	return neutron_cpu_kernel(n, params, absorbed, n, rng);
}
