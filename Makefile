OPTIMIZATION = -O3

CXX = g++
CXXFLAGS = -Wall ${OPTIMIZATION} -lm ${TEST} ${SAVE}

NVCC = nvcc
NVCCFLAGS = -gencode arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60 --compiler-options="-Wall ${OPTIMIZATION} -fopenmp" ${TEST}

LIBS = -lm -L/opt/cuda/lib64/ -lcuda -lcudart

CXX_FILES := main.cpp compaction.cpp

.PHONY: all clean

all: neutron-seq neutron-omp neutron-cud neutron-hyb neutron-ocl neutron-mpi

save: clean
	$(MAKE) SAVE=-DSAVE -C .

# SEQUENTIAL
neutron-seq: Makefile *.cpp *.h
	${CXX} -DNEUTRON_SEQ ${CXXFLAGS} ${CXX_FILES} neutron_seq.cpp -o bin/$@

# OPENMP
neutron-omp: Makefile *.cpp *.h
	${CXX} -DNEUTRON_OMP ${CXXFLAGS} neutron_omp.cpp -fopenmp ${CXX_FILES} neutron_seq.cpp -o bin/$@

# CUDA intermediate files
obj/neutron_cuda_kernel.o: Makefile *.cpp *.h *.cu
	${NVCC} -DNEUTRON_CUD -c neutron_cuda_kernel.cu ${NVCCFLAGS} -o $@ $(LIBS)

obj/neutron_cuda_caller.o: Makefile *.cpp *.h *.cu
	${NVCC} -DNEUTRON_CUD -c neutron_cuda_caller.cu ${NVCCFLAGS} -o $@ $(LIBS)

obj/neutron_gpu_cuda.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_CUD -c ${CXXFLAGS} main.cpp -o $@

obj/neutron_gpu_hyb.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp main.cpp -o $@

obj/neutron_hybrid.o: Makefile *.cpp *.h
	${CXX} -DNEUTRON_HYB -c ${CXXFLAGS} -fopenmp neutron_hybrid.cpp -o $@

# CUDA
neutron-cud: obj/neutron_gpu_cuda.o obj/neutron_cuda_kernel.o obj/neutron_cuda_caller.o
	${NVCC} -DNEUTRON_CUD neutron_gpu.cpp $^ ${NVCCFLAGS} -o bin/$@

# HYBRID : CUDA + OPENMP
neutron-hyb: obj/neutron_hybrid.o obj/neutron_gpu_hyb.o obj/neutron_cuda_kernel.o obj/neutron_cuda_caller.o
	${NVCC} -DNEUTRON_HYB neutron_omp.cpp compaction.cpp neutron_gpu.cpp $^ ${NVCCFLAGS} -o bin/$@ $(LIBS) -lgomp

# OPENCL intermediate files
obj/neutron_opencl_caller.o: Makefile *.cpp *.h *.cl
	${CXX} -DNEUTRON_OCL -c neutron_opencl_caller.cpp ${CXXFLAGS} -lOpenCL -o $@

obj/neutron_gpu_ocl.o: obj/neutron_opencl_caller.o
	${CXX} -DNEUTRON_OCL -c neutron_gpu.cpp ${CXXFLAGS} -lOpenCL -o $@

# OPENCL
neutron-ocl: obj/neutron_gpu_ocl.o obj/neutron_opencl_caller.o
	${CXX} -DNEUTRON_OCL ${CXXFLAGS} -lOpenCL ${CXX_FILES} $^ -o bin/$@

# MPI (with GPU)
obj/neutron_mpi.o: obj/neutron_gpu_cuda.o Makefile *.cpp *.h *.cl
	mpic++ -Wall -DNEUTRON_MPI -c neutron_mpi.cpp ${CXXFLAGS} -o $@

neutron-mpi: obj/neutron_mpi.o obj/neutron_cuda_kernel.o obj/neutron_cuda_caller.o
	${NVCC} -DNEUTRON_MPI ${NVCCFLAGS} -ccbin=mpic++ ${CXX_FILES} neutron_gpu.cpp $^ -o bin/$@

test: clean
	$(MAKE) TEST+=-DTEST -C .
	./bin/neutron-seq 1.0 100000000 0.5 0.5 | grep "TEST"
	./bin/neutron-omp 1.0 100000000 0.5 0.5 | grep "TEST"
	./bin/neutron-cud 1.0 100000000 0.5 0.5 | grep "TEST"
	./bin/neutron-hyb 1.0 100000000 0.5 0.5 32 20000 0.1 | grep "TEST"
	./bin/neutron-ocl 1.0 100000000 0.5 0.5 32 20000 GPU | grep "TEST"
	./bin/neutron-mpi 1.0 100000000 0.5 0.5 32 20000 | grep -a "TEST"

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

cud_bench: neutron-cud
	./bin/neutron-cud 1.0 500000000 0.1 0.9 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.2 0.8 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.3 0.7 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.4 0.6 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.5 0.5 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.6 0.4 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.7 0.3 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.8 0.2 32 20000 | grep Millions
	./bin/neutron-cud 1.0 500000000 0.9 0.1 32 20000 | grep Millions

clean:
	rm -f bin/* obj/*.o *~
