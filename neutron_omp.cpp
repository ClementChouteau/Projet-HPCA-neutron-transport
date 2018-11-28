#include "neutron_omp.h"

#include "compaction.h"
#include "neutron_cpu_kernel.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>

using namespace std::chrono;

ExperimentalResults neutron_omp(float* absorbed, int n, const ProblemParameters& params) {
	const auto BLOCK_SIZE = 10000;

	ExperimentalResults res;
	res.r = 0;
	res.b = 0;
	res.t = 0;
	res.absorbed = absorbed;

	std::vector<ExperimentalResults> to_reduce;

	std::vector<std::mt19937> rngs;
	#pragma omp parallel
	#pragma omp single
	{
		// Initializing random generators
		auto t = 1;
		#ifdef _OPENMP
		t = omp_get_num_threads();
		#endif
		rngs = std::vector<std::mt19937>(t);
		for (auto& rng : rngs)
			rng.seed(std::random_device()());

		int i=0;
		while (i<n) {
			// Distributing work (that we know)
			while (i<n) {
				int to_do;
				ExperimentalResults cur_res;
				int buffer_size = BLOCK_SIZE;
				#pragma omp critical (scheduling)
				{
					// Work to be done
					const int work_remainder = n-i;
					to_do = (work_remainder>=BLOCK_SIZE) ? BLOCK_SIZE : work_remainder;

					// Buffer to write results
					if (to_reduce.empty()) {
						cur_res.r = 0;
						cur_res.b = 0;
						cur_res.t = 0;
						cur_res.absorbed = absorbed;
						absorbed += BLOCK_SIZE;
					}
					else {
						cur_res = to_reduce.back();
						to_reduce.pop_back();
						buffer_size = BLOCK_SIZE - cur_res.b;
					}
				}

				#pragma omp task shared(i, rngs, res, to_reduce) firstprivate(to_do, params, cur_res, buffer_size) default(none)
				{
					int num = 0;
					#ifdef _OPENMP
					num = omp_get_thread_num();
					#endif
					auto& rng = rngs[num];
					const auto call_res = neutron_cpu_kernel(to_do, params, cur_res.absorbed, buffer_size, rng);

					// Given work done partially
					auto done = call_res.r+call_res.b+call_res.t;

					#pragma omp atomic // assume i = i + to_do
					i -= to_do - done; // then   i = i + to_do - (to_do - done)

					cur_res.r += call_res.r;
					cur_res.b += call_res.b;
					cur_res.t += call_res.t;
					cur_res.absorbed += call_res.b;

					// Buffer complete
					if (cur_res.b == BLOCK_SIZE) {
						#pragma omp atomic
						res.r += cur_res.r;
						#pragma omp atomic
						res.b += cur_res.b;
						#pragma omp atomic
						res.t += cur_res.t;
						// ...discard it
					}
					// Buffer incomplete
					else
						#pragma omp critical(scheduling)
						to_reduce.push_back(cur_res);
				}

				// Assume the task did the work intended
				// (Exact count will be decreased in task)
				#pragma omp atomic
				i += to_do;
			}

			// Wait for work restitution
			#pragma omp taskwait
		}
	}

	const auto start = std::chrono::system_clock::now();
	auto compare_by_addr = [](const ExperimentalResults& l, const ExperimentalResults& r)
	{
		return l.absorbed < r.absorbed;
	};
	std::sort(to_reduce.begin(), to_reduce.end(), compare_by_addr);
	compaction(res, std::vector<int>(to_reduce.size(), BLOCK_SIZE), to_reduce);
	const auto finish = std::chrono::system_clock::now();

	std::cout << "Temps réduction post OpenMP task: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()/1000. << " sec" << std::endl;

	return res;
}
