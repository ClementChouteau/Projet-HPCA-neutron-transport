/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/remove.h>

#if __has_include ( "thrust/device_ptr.h" )
#include <thrust/device_ptr.h>
#endif

#include "neutron.h"

#include <utility>
#include <chrono>

#include <stdio.h>

#include "neutron_gpu_kernel.h"

using namespace std::chrono;

/**
 * Retourne le quotient entier superieur ou egal a "a/b".
 */
template<typename T>
inline static T iDivUp(T a, T b){
	return ((a % b != 0) ? (a / b + 1) : (a / b));
}

ExperimentalResults neutron_gpu_caller(float* absorbed, long n,
																			 const ProblemParameters& params,
																			 const std::vector<unsigned long long>& seeds,
																			 int threadsPerBlock, int neutronsPerThread) {
	const auto threads = threadsPerBlock*iDivUp<long>(n, threadsPerBlock*neutronsPerThread);

	auto t1 = system_clock::now();
	unsigned long long* d_seeds;
	cudaMalloc((void**)&d_seeds, seeds.size()*sizeof(unsigned long long));
	cudaMemcpy(d_seeds, seeds.data(), seeds.size()*sizeof(unsigned long long), cudaMemcpyHostToDevice);

	// launching cuda kernel
	ProblemParameters* d_params;
	cudaMalloc((void**)&d_params, sizeof(ProblemParameters));
	cudaMemcpy(d_params, &params, sizeof(ProblemParameters), cudaMemcpyHostToDevice);

	float* d_absorbed;
	cudaMalloc((void**)&d_absorbed, n*sizeof(float));

	long* d_r, * d_b, * d_t;
	cudaMalloc((void**)&d_r, threads*sizeof(long));
	cudaMalloc((void**)&d_b, threads*sizeof(long));
	cudaMalloc((void**)&d_t, threads*sizeof(long));

	curandState* d_states;
	cudaMalloc((void**)&d_states, threads*sizeof(curandState));
	auto t2 = system_clock::now();
	std::cout << "Temps de la copie CPU -> GPU: " << std::chrono::duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	const dim3 nthreads(threadsPerBlock);
	const dim3 nblocs(iDivUp<long>(n, threadsPerBlock*neutronsPerThread));
	std::cout << "Nombre de blocs GPU: " << nblocs.x << std::endl;
	std::cout << "Nombre de threads par bloc: " << nthreads.x << std::endl;
	std::cout << "Mémoire utilisée: " << (n*4.)/(1024.*1024.) << "Mo" << std::endl;

	auto t3 = system_clock::now();
	neutron_seq_kernel<<<nthreads, nblocs>>>(n, neutronsPerThread, d_params, d_absorbed,
																					 d_r, d_b, d_t, d_seeds, d_states);
	cudaFree(d_seeds);

	// reductions
	thrust::remove_if(thrust::device_ptr<float>(d_absorbed),
										thrust::device_ptr<float>(d_absorbed) + n,
										thrust::placeholders::_1 == NO_VAL);

	ExperimentalResults res;
	res.r = thrust::reduce(thrust::device, d_r, d_r + threads);
	res.b = thrust::reduce(thrust::device, d_b, d_b + threads);
	res.t = thrust::reduce(thrust::device, d_t, d_t + threads);

	// retrieving results
	cudaDeviceSynchronize();
	auto t4 = system_clock::now();
	std::cout << "Temps du kernel + reductions: " << std::chrono::duration_cast<milliseconds>(t4 - t3).count()/1000. << " sec" << std::endl;

	cudaFree(d_r);
	cudaFree(d_b);
	cudaFree(d_t);

	if (res.r+res.b+res.t != n)
		exit(1);

	t1 = system_clock::now();
	res.absorbed = absorbed;
	cudaMemcpy(res.absorbed, d_absorbed, res.b*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_absorbed);
	t2 = system_clock::now();
	std::cout << "Temps de la copie GPU -> CPU: " << std::chrono::duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	return res;
}
