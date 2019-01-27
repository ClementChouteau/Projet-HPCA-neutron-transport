#ifndef NEUTRON_OMP_H
#define NEUTRON_OMP_H

#include "neutron.h"

ExperimentalResults neutron_omp(float* absorbed, long n,
																const ProblemParameters& params);

#endif // NEUTRON_OMP_H
