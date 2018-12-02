#ifndef COMPACTION_H
#define COMPACTION_H

#include <vector>

#include "neutron.h"

void compaction(ExperimentalResults& res, const std::vector<long> sizes,
								const std::vector<ExperimentalResults>& to_reduce);

#endif // COMPACTION_H
