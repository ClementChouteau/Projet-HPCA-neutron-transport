#ifndef NEUTRON_MPI_H
#define NEUTRON_MPI_H

#include <mpi.h>

#include "neutron.h"

ExperimentalResults neutron_mpi(
		float* absorbed,
		long N,
		const ProblemParameters& params,
		int threadsPerBlock,
		int neutronsPerThread,
		int mpiBlockSize
);

#endif // NEUTRON_MPI_H
