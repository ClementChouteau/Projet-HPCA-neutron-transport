#include "neutron_opencl_caller.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <iostream>
#include <vector>

/**
 * Retourne le quotient entier superieur ou egal a "a/b".
 */
template<typename T>
inline static T iDivUp(T a, T b){
	return ((a % b != 0) ? (a / b + 1) : (a / b));
}

// Adapted from OpenCL example: https://gist.github.com/ddemidov/2925717

ExperimentalResults neutron_opencl_caller(
		float* absorbed,
		long n,
		const ProblemParameters& params,
		const std::vector<unsigned long long>& ullseeds,
		int threadsPerBlock,
		int neutronsPerThread,
		std::string oclDeviceType
) {

	const std::vector<unsigned long> seeds(ullseeds.begin(), ullseeds.end());

	// Get list of OpenCL platforms.
	std::vector<cl::Platform> platform;
	cl::Platform::get(&platform);

	if (platform.empty()) {
		std::cerr << "OpenCL platforms not found." << std::endl;
		exit(1);
	}

	// Get first available GPU device
	cl::Context context;
	std::vector<cl::Device> device;
	for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
		std::vector<cl::Device> pldev;

		if (oclDeviceType == "CPU")
			p->getDevices(CL_DEVICE_TYPE_CPU, &pldev);
		else if (oclDeviceType == "GPU")
			p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);
		else {
			std::cerr << "(neutron_opencl_caller) unknown option : " << oclDeviceType << std::endl;
			exit(0);
		}

		for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
			if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

			device.push_back(*d);
			context = cl::Context(device);
			goto device_found;
		}
	}
	device_found:

	if (device.empty()) {
		std::cerr << "GPU not found !" << std::endl;
		exit(1);
	}

	std::cout << "Device found: " << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

	// Compile OpenCL program for found device.
	const char *source =
		#include "neutron_opencl_kernel.cl"
	;

	cl::Program program(context, cl::Program::Sources(
				1, std::make_pair(source, strlen(source))));

	try {
			program.build(device);
	} catch (const cl::Error&) {
			std::cerr
		<< "OpenCL compilation error" << std::endl
		<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
		<< std::endl;
			exit(1);
	}

	// Create command queue.
	cl::CommandQueue queue(context, device[0]);
	cl::Kernel kernel(program, "neutron_opencl_kernel");

	// Allocate device buffers and transfer input data to device.
	cl::Buffer d_params(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		1 * sizeof(ProblemParameters), const_cast <ProblemParameters*> (&params));

	unsigned int next_absorbed = 0;
	cl::Buffer d_next_absorbed(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		1 * sizeof(unsigned int), &next_absorbed);

#ifdef TEST
	cl::Buffer d_absorbed(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		n * sizeof(float), absorbed);
#else
	cl::Buffer d_absorbed(context, CL_MEM_WRITE_ONLY,
		n * sizeof(float));
#endif

	unsigned int r = 0, b = 0, t = 0;
	cl::Buffer d_r(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		1 * sizeof(unsigned int), &r);

	cl::Buffer d_b(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		1 * sizeof(unsigned int), &b);

	cl::Buffer d_t(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		1 * sizeof(unsigned int), &t);

	cl::Buffer d_seeds(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		seeds.size() * sizeof(unsigned long), const_cast <unsigned long*> (seeds.data()));

	// Set kernel parameters.
	kernel.setArg(0, static_cast<cl_uint>(n));
	kernel.setArg(1, static_cast<cl_uint>(neutronsPerThread));
	kernel.setArg(2, d_params);
	kernel.setArg(3, d_next_absorbed);
	kernel.setArg(4, d_absorbed);
	kernel.setArg(5, d_r);
	kernel.setArg(6, d_b);
	kernel.setArg(7, d_t);
	kernel.setArg(8, d_seeds);

	// Launch kernel on the compute device.
	const auto localWorkSize = cl::NDRange(threadsPerBlock);
	const auto globalWorkSize = cl::NDRange(threadsPerBlock*iDivUp<long>(n, neutronsPerThread*threadsPerBlock));
	try {
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, localWorkSize);
	} catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		exit(1);
	}

	// Get result back to host.
	queue.enqueueReadBuffer(d_r, CL_TRUE, 0, 1*sizeof(r), &r);
	queue.enqueueReadBuffer(d_b, CL_TRUE, 0, 1*sizeof(b), &b);
	queue.enqueueReadBuffer(d_t, CL_TRUE, 0, 1*sizeof(t), &t);
	ExperimentalResults res;
	res.r = static_cast<long>(r);
	res.b = static_cast<long>(b);
	res.t = static_cast<long>(t);
	res.absorbed = absorbed;
	queue.finish();

	queue.enqueueReadBuffer(d_absorbed, CL_TRUE, 0, res.b * sizeof(float), res.absorbed);
	queue.finish();

	return res;
}
