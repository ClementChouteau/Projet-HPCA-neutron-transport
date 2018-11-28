#include "neutron_gpu.h"

#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

using namespace std::chrono;

extern ExperimentalResults neutron_gpu_caller(float* absorbed, int n, const ProblemParameters& params,
																							const std::vector<unsigned long long>& seeds,
																							int threadsPerBlock, int neutronsPerThread);

ExperimentalResults neutron_gpu(float* absorbed, int n,
																const ProblemParameters& params,
																int threadsPerBlock, int neutronsPerThread) {
	const auto threads = n/neutronsPerThread;

	// generating seeds for GPU
	auto t1 = system_clock::now();
	auto seeds = std::vector<unsigned long long>(threads);
	std::iota(seeds.begin(), seeds.end(), 0);

	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<unsigned long long> seedGen;

	for (int i=0; i<threads; i++)
		seeds[i] = seedGen(rng);

	auto t2 = system_clock::now();
	std::cout << "Temps de génération des graines: " << duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	ExperimentalResults res = neutron_gpu_caller(absorbed, n, params, seeds, threadsPerBlock, neutronsPerThread);

	return res;
}
