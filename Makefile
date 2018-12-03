CXX = g++
CXXFLAGS = -Wall -O3 -lm ${TEST} ${SAVE}

NVCC = nvcc
NVCCFLAGS = --compiler-options="-Wall -O3 -fopenmp" ${TEST}

LIBS = -lm -L/opt/cuda/lib64/ -lcuda -lcudart

CXX_FILES := main.cpp compaction.cpp

.PHONY: all clean

all: neutron-seq neutron-omp neutron-gpu neutron-hybrid

save: clean
	$(MAKE) SAVE=-DSAVE -C .

# SEQUENTIAL
neutron-seq: Makefile *.cpp *.h
	${CXX} -DNEUTRON_SEQ ${CXXFLAGS} ${CXX_FILES} neutron_seq.cpp -o bin/$@

# OPENMP
neutron-omp: Makefile *.cpp *.h
	${CXX} -DNEUTRON_OMP ${CXXFLAGS} neutron_omp.cpp -fopenmp ${CXX_FILES} neutron_seq.cpp -o bin/$@

# CUDA intermediate files
obj/neutron_gpu_kernel.o: Makefile *.cpp *.h *.cu
	${NVCC} -c neutron_gpu_kernel.cu ${NVCCFLAGS} -o $@ $(LIBS)

obj/neutron_gpu_caller.o: Makefile *.cpp *.h *.cu
	${NVCC} -c neutron_gpu_caller.cu ${NVCCFLAGS} -o $@ $(LIBS)

obj/neutron_gpu_gpu.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_GPU -c ${CXXFLAGS} main.cpp -o $@

obj/neutron_gpu_omp.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp main.cpp -o $@

obj/neutron_hybrid.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp neutron_hybrid.cpp -o $@

# GPU
neutron-gpu: obj/neutron_gpu_gpu.o obj/neutron_gpu_kernel.o obj/neutron_gpu_caller.o
	${NVCC} neutron_gpu.cpp $^ ${NVCCFLAGS} -o bin/$@

# HYBRID : GPU + OPENMP
neutron-hybrid: obj/neutron_hybrid.o obj/neutron_gpu_omp.o obj/neutron_gpu_kernel.o obj/neutron_gpu_caller.o
	${NVCC} neutron_omp.cpp compaction.cpp neutron_gpu.cpp $^ ${NVCCFLAGS} -o bin/$@ $(LIBS) -lgomp

test: clean
	$(MAKE) TEST+=-DTEST -C .
	./bin/neutron-seq 1.0 500000000 0.5 0.5 | grep "TEST"
	./bin/neutron-omp 1.0 500000000 0.5 0.5 | grep "TEST"
	./bin/neutron-gpu 1.0 500000000 0.5 0.5 | grep "TEST"
	./bin/neutron-hybrid 1.0 500000000 0.5 0.5 32 20000 0.1 | grep "TEST"

omp_bench: neutron-omp
	./bin/neutron-omp 1.0 500000000 0.1 0.9 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.2 0.8 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.3 0.7 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.4 0.6 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.5 0.5 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.6 0.4 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.7 0.3 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.8 0.2 32 20000 | grep Millions
	./bin/neutron-omp 1.0 500000000 0.9 0.1 32 20000 | grep Millions

gpu_bench: neutron-gpu
	./bin/neutron-gpu 1.0 500000000 0.1 0.9 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.2 0.8 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.3 0.7 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.4 0.6 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.5 0.5 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.6 0.4 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.7 0.3 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.8 0.2 32 20000 | grep Millions
	./bin/neutron-gpu 1.0 500000000 0.9 0.1 32 20000 | grep Millions

clean:
	rm -f bin/* obj/*.o *~
