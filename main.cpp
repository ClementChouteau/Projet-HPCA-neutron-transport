/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef SAVE
#include <fstream>
#endif

#include "neutron.h"
#if   defined (NEUTRON_SEQ)
#include "neutron_seq.h"
#elif defined (NEUTRON_OMP)
#include "neutron_omp.h"
#elif defined (NEUTRON_GPU)
#include "neutron_gpu.h"
#elif defined (NEUTRON_HYB)
#include "neutron_hybrid.h"
#endif

using namespace std::chrono;

#define OUTPUT_FILE "/tmp/absorbed.dat"

const static char info[] =
		"\
		Usage:\n\
		neutron-seq H Nb C_c C_s\n\
		\n\
		H  : épaisseur de la plaque\n\
		Nb : nombre d'échantillons\n\
		C_c: composante absorbante\n\
		C_s: componente diffusante\n\
		(t): neutrons par thread\n\
		(b): threads par block\n\
		\n\
		Exemple d'execution : \n\
		neutron-seq 1.0 500000000 0.5 0.5 32 10000\n\
		"
;

int main(int argc, char *argv[]) {
	std::ios_base::sync_with_stdio(false);

	if(argc == 1) {
		std::cerr << info << std::endl;
		exit(0);
	}

	// Default values
	ProblemParameters params;
	params.h = 1.0;
	long n = 500000000; // number of neutrons
	params.c_c = 0.5;
	params.c_s = 0.5;

	int threadsPerBlock = 32;
	(void) threadsPerBlock;
	int neutronsPerThread = 20000;
	(void) neutronsPerThread;

	float ratio = 1.0;
	(void) ratio;

	// Retrieving parameters
	if (argc > 1) params.h = std::atof(argv[1]);
	if (argc > 2) n = std::atol(argv[2]);
	if (argc > 3) params.c_c = std::atof(argv[3]);
	if (argc > 4) params.c_s = std::atof(argv[4]);
	if (argc > 5) threadsPerBlock = std::atoi(argv[5]);
	if (argc > 6) neutronsPerThread = std::atoi(argv[6]);
	if (argc > 7) ratio = std::atof(argv[7]);

	params.c = params.c_c + params.c_s;

	// Printing parameters (for checking)
	std::cout << "Épaisseur de la plaque : " << params.h << std::endl;
	std::cout << "Nombre d'échantillons  : " << n << std::endl;
	std::cout << "C_c : " << params.c_c << std::endl;
	std::cout << "C_s : " << params.c_s << std::endl;

	std::vector<float> absorbed(n);
#ifdef TEST
	std::fill(absorbed.begin(), absorbed.end(), NO_VAL);
#endif

	const auto start = system_clock::now();
#if   defined (NEUTRON_SEQ)
	const ExperimentalResults res = neutron_seq(absorbed.data(), n, params);
#elif defined (NEUTRON_OMP)
	const ExperimentalResults res = neutron_omp(absorbed.data(), n, params);
#elif defined (NEUTRON_GPU)
	const ExperimentalResults res = neutron_gpu(absorbed.data(), n, params, threadsPerBlock, neutronsPerThread);
#elif defined (NEUTRON_HYB)
	const ExperimentalResults res = neutron_hybrid(absorbed.data(), n, params, threadsPerBlock, neutronsPerThread, ratio);
#endif
	const auto finish = system_clock::now();

#ifdef TEST
	long b = n - std::count(absorbed.begin(), absorbed.end(), NO_VAL);
	if (b != res.b) {
		std::cout << "TEST FAILURE" << std::endl;
		exit(1);
	}
	std::cout << "TEST SUCESS" << std::endl;
#endif

	if (res.r+res.b+res.t != n)
		exit(1);

	std::cout << std::endl;
	std::cout << "Pourcentage des neutrons refléchis : " << (double) res.r / (double) n << std::endl;
	std::cout << "Pourcentage des neutrons absorbés : " << (double) res.b / (double) n << std::endl;
	std::cout << "Pourcentage des neutrons transmis : " << (double) res.t / (double) n << std::endl;

	const auto duration = duration_cast<milliseconds>(finish - start).count()/1000.;
	std::cout << std::endl;
	std::cout << "Temps total de calcul: " << duration << " sec" << std::endl;
	std::cout << "Millions de neutrons /s: " << (double) n / ((duration)*1e6) << std::endl;

#ifdef SAVE
	std::ofstream file(OUTPUT_FILE);
	if (file.is_open()) {
		for (int i=0; i<res.b; i++)
			file << res.absorbed[i] << std::endl;
	}
	else {
		std::cerr << "Cannot open " << OUTPUT_FILE << std::endl;
		exit(1);
	}

	std::cout << "Result written in " << OUTPUT_FILE << std::endl;
#endif

	return 0;
}

