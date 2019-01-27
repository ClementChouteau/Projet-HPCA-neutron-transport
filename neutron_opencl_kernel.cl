R"(
#define NO_VAL (-1)

typedef struct {
	// La distance moyenne entre les interactions neutron/atome est 1/c.
	// c_c et c_s sont les composantes absorbantes et diffusantes de c.
	float c, c_c, c_s;
	// épaisseur de la plaque
	float h;
	// distance parcourue par le neutron avant la collision
	float L;
} ProblemParameters;

typedef struct {
	// nombre de neutrons refléchis, absorbés et transmis
	long r, b, t;

	float *absorbed;
} ExperimentalResults;

//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

// https://en.wikipedia.org/wiki/Xorshift
inline float xorshift64(ulong* state)
{
	ulong x = *state;

	x^= x << 13;
	x^= x >> 7;
	x^= x << 17;

	*state = x;

	return (float)x/(float)ULONG_MAX;
}


void kernel neutron_opencl_kernel
(
		const uint n,
		const uint neutronsPerThread,
		global const ProblemParameters* params,
		volatile global uint* next_absorbed,
		volatile global float* absorbed,
		volatile global uint* d_r,
		volatile global uint* d_b,
		volatile global uint* d_t,
		global const ulong* seeds
) {
	const uint id = get_global_id(0);

	ulong state = seeds[id];

	const float c   = params->c;
	const float c_c = params->c_c;
	const float h   = params->h;

	uint r = 0, b = 0, t = 0; // int is enough for local counts

	const uint k = id*neutronsPerThread;
	const uint m = max( min((long) neutronsPerThread, (long) n-k), (long)0);
	for (uint i=0; i<m; i++) {

		float d = 0.0;
		float x = 0.0;

		float v;
		while (1) {

			const float u = xorshift64(&state);
			const float L = -(1 / c) * log(u);
			x = x + L * cos(d);

			v = NO_VAL;
			if (x < 0) {
				r++;
				break;
			}
			else if (x >= h) {
				t++;
				break;
			}
			else if (xorshift64(&state) < c_c / c) {
				b++;
				v = x;
				break;
			}
			else {
				const float u = xorshift64(&state);
				d = u * M_PI;
			}
		}

		if (v != NO_VAL) {
			uint pos = atomic_inc(next_absorbed);
			absorbed[pos] = v;
		}
	}

	atomic_add(d_r, r);
	atomic_add(d_b, b);
	atomic_add(d_t, t);
}
)"

