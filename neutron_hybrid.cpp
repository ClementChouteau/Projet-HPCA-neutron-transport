#include "neutron_hybrid.h"

#include "compaction.h"
#include "neutron_omp.h"
#include "neutron_gpu.h"

#include <iostream>
#include <vector>
#include <chrono>

using namespace std::chrono;

ExperimentalResults neutron_hybrid(float* absorbed, int n,
																	 const ProblemParameters& params,
																	 int threadsPerBlock, int neutronsPerThread,
																	 float ratio) {
	const int n_cpu = n*ratio;
	const int n_gpu = n-n_cpu;

	std::cout << "Nombre de neutrons sur CPU: " << n_cpu << std::endl;
	std::cout << "Nombre de neutrons sur GPU: " << n_gpu << std::endl;

	decltype(system_clock::now()) start1, start2, finish1, finish2;

	ExperimentalResults res1, res2;
	#pragma omp parallel
	#pragma omp single
	{
		#pragma omp task
		{
			start1 = system_clock::now();
			res1 = neutron_gpu(absorbed, n_gpu, params, threadsPerBlock, neutronsPerThread);
			finish1 = system_clock::now();
		}

		#pragma omp task
		{
			start2 = system_clock::now();
			res2 = neutron_omp(absorbed+n_gpu, n_cpu, params);
			finish2 = system_clock::now();
		}
	}

	std::cout << "Temps hybride, partie GPU: " << duration_cast<milliseconds>(finish1 - start1).count()/1000. << " sec" << std::endl;
	std::cout << "Temps hybride, partie CPU: " << duration_cast<milliseconds>(finish2 - start2).count()/1000. << " sec" << std::endl;

	// Compaction of the CPU+GPU results
	auto start3 = system_clock::now();
	ExperimentalResults res;
	res.r = 0;
	res.b = 0;
	res.t = 0;
	res.absorbed = absorbed;

	std::vector<ExperimentalResults> to_reduce({res1, res2});
	std::vector<int> sizes({n_gpu, n_cpu});
	compaction(res, sizes, to_reduce);
	auto finish3 = system_clock::now();

	std::cout << "Temps hybride, reduction: " << duration_cast<milliseconds>(finish3 - start3).count()/1000. << " sec" << std::endl;

	return res;
}
