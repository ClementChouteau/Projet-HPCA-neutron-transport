#include "neutron_mpi.h"

#include "neutron_gpu.h"

#include <iostream>
#include <vector>
#include <list>

ExperimentalResults neutron_mpi(
		float* absorbed,
		long N,
		const ProblemParameters& params,
		int threadsPerBlock,
		int neutronsPerThread,
		int mpiBlockSize)
{
	ExperimentalResults res;
	res.absorbed = absorbed;
	res.r = 0;
	res.b = 0;
	res.t = 0;

	MPI_Init(NULL, NULL);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int n = 0;

	// Slave : work then send results async
	if (rank != 0) {
		// Disable console outputs
		std::cout.rdbuf(NULL);
		std::cerr.rdbuf(NULL);

		std::list<ExperimentalResults> results;
		std::list<MPI_Request> reqs_absorbed;
		while (1) {
			// More work to do ?
			int ok;
			MPI_Recv(&ok, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (!ok) break;

			results.push_back( neutron_gpu(absorbed, mpiBlockSize, params, threadsPerBlock, neutronsPerThread) );

			// Send counts
			MPI_Send(&(results.back().r), 3, MPI_LONG, 0, 0, MPI_COMM_WORLD);

			// Send neutrons
			reqs_absorbed.push_back(MPI_REQUEST_NULL);
			MPI_Isend(absorbed,
								results.back().b,
								MPI_FLOAT,
								0,
								0,
								MPI_COMM_WORLD,
								&reqs_absorbed.back());

			absorbed += res.b;
		}

		// Wait end of sends
		for (MPI_Request& r : reqs_absorbed) {
			MPI_Wait(&r, MPI_STATUS_IGNORE);
		}

		MPI_Finalize();
		exit(0);
	}

	// Master : initiate receives, end the work
	else {
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		// Initiate reception of the counts
		// [0] of these arrays will be unused
		std::vector<long> r_b_t(3*world_size, 0);
		std::vector<MPI_Request> reqs_r_b_t(world_size, MPI_REQUEST_NULL);
		for (int i=1; i<world_size; i++) {
			const int ok = (n<N-mpiBlockSize);
			MPI_Send(&ok, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			if (ok) {
				MPI_Irecv(&r_b_t[3*i], 3, MPI_LONG, i, 0, MPI_COMM_WORLD, &reqs_r_b_t[i]);
				n += mpiBlockSize;
			}
		}

		// When a count is received, initiate reception of neutrons
		std::list<MPI_Request> reqs_absorbed;
		int i;
		while (MPI_Waitany(world_size, reqs_r_b_t.data(), &i, MPI_STATUS_IGNORE) == MPI_SUCCESS) {
			if (i == MPI_UNDEFINED) break;
			reqs_r_b_t[i] = MPI_REQUEST_NULL;

			long& r = r_b_t[3*i+0];
			long& b = r_b_t[3*i+1];
			long& t = r_b_t[3*i+2];
			res.r += r;
			res.b += b;
			res.t += t;

			// Receive neutrons
			reqs_absorbed.push_back(MPI_REQUEST_NULL);
			MPI_Irecv(absorbed, b, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &reqs_absorbed.back());
			absorbed += b;

			r = 0;
			b = 0;
			t = 0;

			// Send (more work or not ?)
			const int ok = (n<N-mpiBlockSize);
			MPI_Send(&ok, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

			// Ask for more work
			if (ok) {
				MPI_Irecv(&r_b_t[3*i], 3, MPI_LONG, i, 0, MPI_COMM_WORLD, &reqs_r_b_t[i]);
				n += mpiBlockSize;
			}
		}

		// End work
		if (n != N) {
			const auto res2 = neutron_gpu(absorbed, N-n, params, threadsPerBlock, neutronsPerThread);
			res.r += res2.r;
			res.b += res2.b;
			res.t += res2.t;
		}
	}

	MPI_Finalize();

	return res;
}
