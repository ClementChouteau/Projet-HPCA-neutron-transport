CXX = g++
CXXFLAGS = -Wall -O3 -lm

NVCC=nvcc

LIBS = -lm -L/opt/cuda/lib64/ -lcuda -lcudart

CXX_FILES := main.cpp compaction.cpp

all: neutron-seq neutron-omp neutron-gpu neutron-hybrid

# SEQUENTIAL
neutron-seq: Makefile *.cpp *.h
	${CXX} -DNEUTRON_SEQ ${CXXFLAGS} ${CXX_FILES} neutron_seq.cpp -o bin/$@

# OPENMP
neutron-omp: Makefile *.cpp *.h
	${CXX} -DNEUTRON_OMP ${CXXFLAGS} neutron_omp.cpp -fopenmp ${CXX_FILES} neutron_seq.cpp -o bin/$@

# CUDA intermediate files
obj/neutron_gpu_kernel.o: Makefile *.cpp *.h *.cu
	${NVCC} -c neutron_gpu_kernel.cu -o $@ $(LIBS)

obj/neutron_gpu_caller.o: Makefile *.cpp *.h *.cu
	${NVCC} -c neutron_gpu_caller.cu -o $@ $(LIBS)

obj/neutron_gpu_gpu.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_GPU -c ${CXXFLAGS} main.cpp -o $@

obj/neutron_gpu_omp.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp main.cpp -o $@

obj/neutron_hybrid.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp neutron_hybrid.cpp -o $@

# GPU
neutron-gpu: obj/neutron_gpu_gpu.o obj/neutron_gpu_kernel.o obj/neutron_gpu_caller.o
	${NVCC} neutron_gpu.cpp $^ -o bin/$@

# HYBRID : GPU + OPENMP
neutron-hybrid: obj/neutron_hybrid.o obj/neutron_gpu_omp.o obj/neutron_gpu_kernel.o obj/neutron_gpu_caller.o
	${NVCC} neutron_omp.cpp compaction.cpp neutron_gpu.cpp $^ -o bin/$@ $(LIBS) -lgomp

clean:
	rm -f bin/* obj/*.o *~
