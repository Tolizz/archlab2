
#include "xcl2.hpp"
#include "event_timer.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>

#define HEIGHT 128
#define WIDTH 128
#define DATA_SIZE (HEIGHT * WIDTH)
#define T1 32
#define T2 96

// Software reference
int clip_sw(int x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return x;
}

void software_reference(std::vector<int, aligned_allocator<int>>& A,
                        std::vector<int, aligned_allocator<int>>& B,
                        std::vector<int, aligned_allocator<int>>& C_Ref) {

    // intermediate table for Posterize
    std::vector<int> temp_C(DATA_SIZE);

    // 1. Difference & Posterize
    for (int i = 0; i < DATA_SIZE; i++) {
        int diff = A[i] - B[i];
        int abs_diff = std::abs(diff);

        if (abs_diff < T1) temp_C[i] = 0;
        else if (abs_diff < T2) temp_C[i] = 128;
        else temp_C[i] = 255;
    }

    // 2. Sharpen Filter
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int idx = i * WIDTH + j;

            // Boundary Condition
            if (i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1) {
                C_Ref[idx] = temp_C[idx];
            } else {
                int center = temp_C[idx];
                int up     = temp_C[(i - 1) * WIDTH + j];
                int down   = temp_C[(i + 1) * WIDTH + j];
                int left   = temp_C[i * WIDTH + (j - 1)];
                int right  = temp_C[i * WIDTH + (j + 1)];

                // Result: 5*Center - Neighbors
                int val = (5 * center) - up - down - left - right;
                C_Ref[idx] = clip_sw(val);
            }
        }
    }

}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  EventTimer et;

  std::string binaryFile = argv[1];
  size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
  cl_int err;
  cl::Context context;
  cl::Kernel krnl_image;
  cl::CommandQueue q;
  // Allocate Memory in Host Memory
  // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
  // hood user ptr
  // is used if it is properly aligned. when not aligned, runtime had no choice
  // but to create
  // its own host side buffer. So it is recommended to use this allocator if
  // user wish to
  // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page
  // boundary. It will
  // ensure that user buffer is used when user create Buffer/Mem object with
  // CL_MEM_USE_HOST_PTR
  et.add("Allocate Memory in Host Memory");
  std::vector<int, aligned_allocator<int>> source_in1(DATA_SIZE);
  std::vector<int, aligned_allocator<int>> source_in2(DATA_SIZE);
  std::vector<int, aligned_allocator<int>> source_hw_results(DATA_SIZE);
  std::vector<int, aligned_allocator<int>> source_sw_results(DATA_SIZE);
  et.finish();

  // Create the test data
  et.add("Fill the buffers");
  std::generate(source_in1.begin(), source_in1.end(), [](){ return rand() % 256; });
  std::generate(source_in2.begin(), source_in2.end(), [](){ return rand() % 256; });
  std::fill(source_hw_results.begin(), source_hw_results.end(), 0);
  et.finish();

  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  et.add("Load Binary File to Alveo U200");
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl_image = cl::Kernel(program, "IMAGE_DIFF_POSTERIZE", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }
  et.finish();

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  et.add("Allocate Buffer in Global Memory");
  OCL_CHECK(err, cl::Buffer buffer_in1(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in1.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_in2(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     vector_size_bytes, source_in2.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_output(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                     vector_size_bytes, source_hw_results.data(), &err));
  et.finish();

  et.add("Set the Kernel Arguments");
  OCL_CHECK(err, err = krnl_image.setArg(0, buffer_in1));
  OCL_CHECK(err, err = krnl_image.setArg(1, buffer_in2));
  OCL_CHECK(err, err = krnl_image.setArg(2, buffer_output));
  et.finish();

  // Copy input data to device global memory
  et.add("Copy input data to device global memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/));
  et.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel
  et.add("Launch the Kernel");
  OCL_CHECK(err, err = q.enqueueTask(krnl_image));
  et.finish();

  // Copy Result from Device Global Memory to Host Local Memory
  et.add("Copy Result from Device Global Memory to Host Local Memory");
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
  OCL_CHECK(err, err = q.finish());
  et.finish();
  // OPENCL HOST CODE AREA END

  et.add("Software Reference Execution");
  software_reference(source_in1, source_in2, source_sw_results);
  et.finish();

  // Compare the results of the Device to the simulation
  et.add("Compare the results of the Device to the simulation");
  bool match = true;
  for (int i = 0; i < DATA_SIZE; i++) {
    if (source_hw_results[i] != source_sw_results[i]) {
      std::cout << "Error: Result mismatch at index " << std::endl;
      std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                << " Device result = " << source_hw_results[i] << std::endl;
      match = false;
      break;
    }
  }

  et.finish();

  std::cout <<"----------------- Key execution times -----------------" << std::endl;
  et.print();

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
