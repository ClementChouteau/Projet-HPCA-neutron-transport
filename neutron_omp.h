#ifndef NEUTRON_OMP_INCLUDED
#define NEUTRON_OMP_INCLUDED

#include "neutron.h"

ExperimentalResults neutron_omp(float* absorbed, long n,
																const ProblemParameters& params);

#endif
