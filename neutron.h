#ifndef NEUTRON_H
#define NEUTRON_H

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

#endif // NEUTRON_H
