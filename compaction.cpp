#include "compaction.h"

#include <utility>

// assumming that to_reduce are in order
void compaction(ExperimentalResults& res, const std::vector<int> sizes,
								const std::vector<ExperimentalResults>& to_reduce) {

	// Sum of the counts
	for (const auto& cur_res : to_reduce) {
		res.r += cur_res.r;
		res.b += cur_res.b;
		res.t += cur_res.t;
	}

	// Compaction of the neutrons
	const int m = to_reduce.size();
	int i = 0;
	int j = m-1;
	while (i < j) {
		int i0 = to_reduce[i].b; // free place (>=)
		int j0 = to_reduce[j].b; // elements to move are (<)

		while (i0 < sizes[i] && j0 > 0) {
			std::swap(to_reduce[i].absorbed[i0], to_reduce[j].absorbed[j0]);
			i0++;
			j0--;
		}

		if (i0 >= sizes[i])
			i++;

		if (j0 <= 0)
			j--;
	}
}
