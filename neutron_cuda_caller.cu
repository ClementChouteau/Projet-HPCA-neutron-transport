/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */

#include <cuda.h>

#include "neutron.h"

#include <utility>
#include <chrono>
#include <iostream>

#include "neutron_cuda_kernel.h"

using namespace std::chrono;

/**
 * Retourne le quotient entier superieur ou egal a "a/b".
 */
template<typename T>
inline static T iDivUp(T a, T b){
	return ((a % b != 0) ? (a / b + 1) : (a / b));
}

ExperimentalResults neutron_cuda_caller(float* absorbed, long n,
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

	unsigned long long int* d_next_absorbed;
	cudaMalloc((void**)&d_next_absorbed, sizeof(unsigned long long int));
	cudaMemset(d_next_absorbed, 0, sizeof(unsigned long long int));

	float* d_absorbed;
	cudaMalloc((void**)&d_absorbed, n*sizeof(float));

#ifdef TEST
	cudaMemcpy(d_absorbed, absorbed, n*sizeof(float), cudaMemcpyHostToDevice);
#endif

	unsigned long long int* d_r, * d_b, * d_t;
	cudaMalloc((void**)&d_r, sizeof(unsigned long long int));
	cudaMalloc((void**)&d_b, sizeof(unsigned long long int));
	cudaMalloc((void**)&d_t, sizeof(unsigned long long int));
	cudaMemset(d_r, 0, sizeof(unsigned long long int));
	cudaMemset(d_b, 0, sizeof(unsigned long long int));
	cudaMemset(d_t, 0, sizeof(unsigned long long int));
	auto t2 = system_clock::now();
	std::cout << "Temps de la copie CPU -> GPU: " << std::chrono::duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	const dim3 nthreads(threadsPerBlock);
	const dim3 nblocs(iDivUp<long>(n, threadsPerBlock*neutronsPerThread));
	std::cout << "Nombre de blocs GPU: " << nblocs.x << std::endl;
	std::cout << "Nombre de threads par bloc: " << nthreads.x << std::endl;
	std::cout << "Mémoire utilisée: " << (n*4.)/(1024.*1024.) << "Mo" << std::endl;

	t1 = system_clock::now();
	neutron_cuda_kernel<<<nthreads, nblocs>>>(n, neutronsPerThread, d_params,
																						d_next_absorbed, d_absorbed,
																						d_r, d_b, d_t, d_seeds);

	// retrieving results
	cudaDeviceSynchronize();
	t2 = system_clock::now();
	std::cout << "Temps du kernel: " << std::chrono::duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	ExperimentalResults res;
	unsigned long long int r, b, t;
	cudaMemcpy(&r, d_r, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&b, d_b, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&t, d_t, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	res.r = static_cast<long>(r);
	res.b = static_cast<long>(b);
	res.t = static_cast<long>(t);

	t1 = system_clock::now();
	res.absorbed = absorbed;

	cudaMemcpy(res.absorbed, d_absorbed, res.b*sizeof(float), cudaMemcpyDeviceToHost);

	t2 = system_clock::now();
	std::cout << "Temps de la copie GPU -> CPU: " << std::chrono::duration_cast<milliseconds>(t2 - t1).count()/1000. << " sec" << std::endl;

	cudaDeviceReset(); // cudaFree(*)

	return res;
}
