#include "neutron_gpu.h"

#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

#if defined (NEUTRON_CUD)
#define KERNEL_CALLER neutron_cuda_caller
#elif defined (NEUTRON_HYB)
#define KERNEL_CALLER neutron_cuda_caller
#elif defined (NEUTRON_OCL)
#define KERNEL_CALLER neutron_opencl_caller
#elif defined (NEUTRON_MPI)
#define KERNEL_CALLER neutron_cuda_caller
#endif

using namespace std::chrono;

/**
 * Retourne le quotient entier superieur ou egal a "a/b".
 */
template<typename T>
inline static T iDivUp(T a, T b){
	return ((a % b != 0) ? (a / b + 1) : (a / b));
}

extern ExperimentalResults neutron_cuda_caller(float* absorbed, long n, const ProblemParameters& params,
																							const std::vector<unsigned long long>& seeds,
																							int threadsPerBlock, int neutronsPerThread);

extern ExperimentalResults neutron_opencl_caller(float* absorbed, long n, const ProblemParameters& params,
																							const std::vector<unsigned long long>& seeds,
																							int threadsPerBlock, int neutronsPerThread,
																								 std::string oclDeviceType);

ExperimentalResults neutron_gpu(float* absorbed, long n,
																const ProblemParameters& params,
																int threadsPerBlock, int neutronsPerThread,
																std::string oclDeviceType) {
	const auto threads = threadsPerBlock*iDivUp<long>(n, neutronsPerThread*threadsPerBlock);

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

#ifndef NEUTRON_OCL
	ExperimentalResults res = KERNEL_CALLER(absorbed, n, params, seeds, threadsPerBlock, neutronsPerThread);
#else
	ExperimentalResults res = KERNEL_CALLER(absorbed, n, params, seeds, threadsPerBlock, neutronsPerThread, oclDeviceType);
#endif

	return res;
}
