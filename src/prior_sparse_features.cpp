#include "prior_sparse_features.h"

PriorSparseFeatures::PriorSparseFeatures()
{
	m_prior_ids = {
		// Chin (0-16)
		21235,
		26155,
		26822,
		26205,
		43800,
		45793,
		46947,
		47733,
		48187,
		48707,
		49433,
		50573,
		52649,
		29302,
		27982,
		29767,
		33877,

		// Eye browes (17-26)
		38740,
		39668,
		40117,
		40346,
		40568,
		41037,
		41222,
		41451,
		41868,
		42686,

		// Nose (27-35)
		8288,
		8304,
		8314,
		8320,
		6524,
		7429,
		8333,
		9234,
		10136,

		// Eye (36-41)
		1830,
		4018,
		5051,
		5958,
		4804,
		3770,

		// Eye (42-47)
		10602,
		11500,
		12532,
		14343,
		12801,
		11770,

		// Mouth (48-59)
		5522,
		6283,
		7442,
		8345,
		9119,
		10411,
		10942,
		9663,
		8760,
		8374,
		7858,
		6955
	};
}

PriorSparseFeatures& PriorSparseFeatures::get()
{
	static PriorSparseFeatures obj;
	return obj;
}
