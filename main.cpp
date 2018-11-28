/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */
#include <chrono>
#include <iostream>

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
	int n = 500000000; // number of neutrons
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
	if (argc > 2) n = std::atoi(argv[2]);
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

	const auto start = system_clock::now();
#if   defined (NEUTRON_SEQ)
	const ExperimentalResults res = neutron_seq(new float[n], n, params);
#elif defined (NEUTRON_OMP)
	const ExperimentalResults res = neutron_omp(new float[n], n, params);
#elif defined (NEUTRON_GPU)
	const ExperimentalResults res = neutron_gpu(new float[n], n, params, threadsPerBlock, neutronsPerThread);
#elif defined (NEUTRON_HYB)
	const ExperimentalResults res = neutron_hybrid(new float[n], n, params, threadsPerBlock, neutronsPerThread, ratio);
#endif
	const auto finish = system_clock::now();

	if (res.r+res.b+res.t != n)
		exit(1);

	std::cout << std::endl;
	std::cout << "Pourcentage des neutrons refléchis : " << (float) res.r / (float) n << std::endl;
	std::cout << "Pourcentage des neutrons absorbés : " << (float) res.b / (float) n << std::endl;
	std::cout << "Pourcentage des neutrons transmis : " << (float) res.t / (float) n << std::endl;

	const auto duration = duration_cast<milliseconds>(finish - start).count()/1000.;
	std::cout << std::endl;
	std::cout << "Temps total de calcul: " << duration << " sec" << std::endl;
	std::cout << "Millions de neutrons /s: " << (double) n / ((duration)*1e6) << std::endl;

	//  // ouverture du fichier pour ecrire les positions des neutrons absorbés
	//  FILE *f_handle = fopen(OUTPUT_FILE, "w");
	//  if (!f_handle) {
	//	fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
	//	exit(EXIT_FAILURE);
	//  }

	//  for (int j = 0; j < res.b; j++)
	//	fprintf(f_handle, "%f\n", res.absorbed[j]);

	//  // fermeture du fichier
	//  fclose(f_handle);
	//  printf("Result written in " OUTPUT_FILE "\n");

	delete res.absorbed;

	return 0;
}

