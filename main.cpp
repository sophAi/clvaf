// OpenCL tutorial 1

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <ctime>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


cl_program load_program(cl_context context, const char* filename) 
{
  std::ifstream in(filename, std::ios_base::binary);
  if (!in.good()) {
    return 0;
  }
  in.seekg(0, std::ios_base::end);
  size_t length = in.tellg();
  in.seekg(0, std::ios_base::beg);

  std::vector<char> data(length + 1);
  in.read(&data[0], length);
  data[length] = 0;

  const char* source = &data[0];
  cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0); 
  if (program == 0) {
    std::cout << "Something wrong while reading source\n";
    return 0;
  }

  if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
    std::cout << "Something wrong while building source\n";
    return 0;
  }

  return program;
}


int main()
{
  cl_int err;
  cl_uint num;
  err = clGetPlatformIDs(0, 0, &num);
  if (err != CL_SUCCESS) {
    std::cerr << "Unable to get platforms\n";
    return 0;
  }
  std::vector<cl_platform_id> platforms(num);
  err = clGetPlatformIDs(num, &platforms[0], &num);
  if (err != CL_SUCCESS) {
    std::cerr << "Unable to get platform ID\n";
    return 0;
  }

  cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
  cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
  if (context == 0) {
    std::cerr << "Can't create OpenCL context\n";
    return 0;
  }

  size_t cb;
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

  clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
  std::string devname;
  devname.resize(cb);
  clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
  std::cout << "Device: " << devname.c_str() << "\n";

  cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
  if (queue == 0) {
    std::cerr << "Can't create command queue\n";
    clReleaseContext(context);
    return 0;
  }
        
  int header_par_num, cycle;
  char dummy1[2], header_source_type[4], header_par_name[26], header_annotation[81];
  float header_par_real;
  int tcf_num, vec_ndim_fac, vec_total_loop, vec_delta_loop, tcf_window, gpu_cycle, tcf_last_window;
  const int char_max_num = 120;
  int file_exist, I0, I1, I2, I3, I4;
  enum bool_enum {is_false, is_true};
  enum odd_even_enum {is_odd, is_even};
  char read_file_name[char_max_num], write_file_name[char_max_num];
  std::cout << "Please input the binary vector file name (ex: *.ffv)\n";
  std::cin >> read_file_name;
  std::ifstream vector_file;
  file_exist = is_false;
  while (file_exist == is_false) {
    vector_file.open(read_file_name,std::ios::in);
    if (!vector_file.good()) {
      std::cout << "Cannot open file = " << read_file_name << std::endl;
      std::cout << "Please type again or Ctrl+C to quit\n";
      std::cin >> read_file_name;
    } else {
      file_exist = is_true;
    }
  }
  std::cout << "Please input the TCF file name (ex: *.tcf)\n";
  std::cin >> write_file_name;
  std::ofstream tcf_file;
  tcf_file.open(write_file_name,std::ios::out);       
  vector_file >> dummy1 >> header_source_type >> header_par_num;
  for (I0 = 0; I0 < header_par_num; I0++) {
    vector_file >> header_par_name >> header_par_real;
    if (!strncmp(header_par_name,"atom_num",8)) {
      tcf_num = (int)header_par_real;
    }
    if (!strncmp(header_par_name,"file_x_dim",10)) {
      vec_total_loop = (int)header_par_real;
    }
    if (!strncmp(header_par_name,"ndim_fac",8)) {
      vec_ndim_fac = (int)header_par_real;
    }      
  }
  std::cout << "Total loop = " << vec_total_loop << std::endl;
  std::cout << "Total TCF function = " << tcf_num << std::endl;
  std::cout << "Dimension of vector = " << vec_ndim_fac << std::endl;
  vector_file >> header_annotation;
  std::cout << "Annotation = " << header_annotation << std::endl;
  std::cout << "Please input window size of TCF\n";
  std::cin >> tcf_window;
  std::cout << "Start to perform GPU calculation...\n";
  time_t start_time = time(NULL);
  float *odd_buffer = new float [tcf_window * vec_ndim_fac];
  float *even_buffer = new float [tcf_window * vec_ndim_fac];
  float *tcf_buffer = new float [tcf_window];

  for (I0 = 0; I0 < tcf_window; I0++) {
    tcf_buffer[I0] = 0;
  }
  for (I0 = 0; I0 < tcf_window * vec_ndim_fac; I0++) {
    odd_buffer[I0] = 0;
    even_buffer[I0] = 0;
  }
  cl_mem cl_ndim_fac = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &vec_ndim_fac, NULL);
  cl_mem cl_tcf_window = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_int), &tcf_window, NULL);
  cl_mem cl_odd_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * tcf_window * vec_ndim_fac, &odd_buffer[0], NULL);
  cl_mem cl_even_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * tcf_window *vec_ndim_fac, &even_buffer[0], NULL);
  cl_mem cl_tcf_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(cl_float) * tcf_window, &tcf_buffer[0], NULL);
  if (cl_ndim_fac == 0 || cl_tcf_window == 0 || cl_odd_buffer == 0 || cl_even_buffer == 0 || cl_tcf_buffer == 0) {
    std::cerr << "Can't create OpenCL buffer\n";
    clReleaseMemObject(cl_ndim_fac);
    clReleaseMemObject(cl_tcf_window);
    clReleaseMemObject(cl_odd_buffer);
    clReleaseMemObject(cl_even_buffer);
    clReleaseMemObject(cl_tcf_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  }

  gpu_cycle = vec_total_loop / tcf_window;
  tcf_last_window = vec_total_loop % tcf_window;
  int last_gpu_cycle =0;
  if (tcf_last_window != 0) {
    last_gpu_cycle = 1;
  }
  std::cout << "gpu cycle = " << gpu_cycle << std::endl;
  cl_program program = load_program(context, "vaf_kernels.cl");
  if (program == 0) {
    std::cerr << "Can't load or build program\n";
    clReleaseMemObject(cl_ndim_fac);
    clReleaseMemObject(cl_tcf_window);
    clReleaseMemObject(cl_odd_buffer);
    clReleaseMemObject(cl_even_buffer);
    clReleaseMemObject(cl_tcf_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  }
  cl_kernel first_tcf = clCreateKernel(program, "first_tcf", 0);
  cl_kernel odd_tcf = clCreateKernel(program, "odd_tcf", 0);
  cl_kernel even_tcf = clCreateKernel(program, "even_tcf",0);
  if (first_tcf == 0 || odd_tcf == 0 || even_tcf == 0) {
    std::cerr << "Can't load kernel\n";
    clReleaseProgram(program);
    clReleaseMemObject(cl_ndim_fac);
    clReleaseMemObject(cl_tcf_window);
    clReleaseMemObject(cl_odd_buffer);
    clReleaseMemObject(cl_even_buffer);
    clReleaseMemObject(cl_tcf_buffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
  }
  clSetKernelArg(first_tcf, 0, sizeof(cl_mem), &cl_ndim_fac);
  clSetKernelArg(first_tcf, 1, sizeof(cl_mem), &cl_tcf_window);
  clSetKernelArg(first_tcf, 2, sizeof(cl_mem), &cl_odd_buffer);
  clSetKernelArg(first_tcf, 3, sizeof(cl_mem), &cl_tcf_buffer);

  clSetKernelArg(odd_tcf, 0, sizeof(cl_mem), &cl_ndim_fac);
  clSetKernelArg(odd_tcf, 1, sizeof(cl_mem), &cl_tcf_window);
  clSetKernelArg(odd_tcf, 2, sizeof(cl_mem), &cl_odd_buffer);
  clSetKernelArg(odd_tcf, 3, sizeof(cl_mem), &cl_even_buffer);
  clSetKernelArg(odd_tcf, 4, sizeof(cl_mem), &cl_tcf_buffer);

  clSetKernelArg(even_tcf, 0, sizeof(cl_mem), &cl_ndim_fac);
  clSetKernelArg(even_tcf, 1, sizeof(cl_mem), &cl_tcf_window);
  clSetKernelArg(even_tcf, 2, sizeof(cl_mem), &cl_odd_buffer);
  clSetKernelArg(even_tcf, 3, sizeof(cl_mem), &cl_even_buffer);
  clSetKernelArg(even_tcf, 4, sizeof(cl_mem), &cl_tcf_buffer);
  float gpu_time = 0;
  time_t gpu_start_time, gpu_end_time;
  size_t work_size = tcf_window;
  for (I0 = 0; I0 < tcf_num; I0++) {
    std::cout << "Calculating tcf id = " << I0 + 1 << std::endl;
    for (I2 = 0; I2 < tcf_window * vec_ndim_fac; I2++) {   //Load the first_tcf kernel
      vector_file >> odd_buffer[I2];
    }
    gpu_start_time = time(NULL);
    err = clEnqueueWriteBuffer(queue, cl_odd_buffer, CL_TRUE, 0, sizeof(float) * tcf_window * vec_ndim_fac, &odd_buffer[0], 0, 0, 0); 

    err = clEnqueueNDRangeKernel(queue, first_tcf, 1, 0, &work_size, 0, 0, 0, 0);
    gpu_end_time = time(NULL);
    gpu_time += (gpu_end_time - gpu_start_time);
    if (err == CL_SUCCESS) {
      err = clEnqueueReadBuffer(queue, cl_tcf_buffer, CL_TRUE, 0, sizeof(float) * tcf_window, &tcf_buffer[0], 0, 0, 0); 
      std::cout << "Printing the first 100 TCF results..." << std::endl;
      for (I2 = 0; I2 < 100; I2++) {
        std::cout << "I = " << I2 << ", tcf = " << tcf_buffer[I2] << std::endl;
      }
      std::cout << "First loop finished\n";
    } 

    for (I1 = 1; I1 < gpu_cycle; I1++) {
      if (I1 % 2 == 0) {        
        cycle = is_odd;
        for (I2 = 0; I2 < tcf_window * vec_ndim_fac; I2++) {
          vector_file >> odd_buffer[I2];
        }
        gpu_start_time = time(NULL);
        err = clEnqueueWriteBuffer(queue, cl_odd_buffer, CL_TRUE, 0, sizeof(float) * tcf_window * vec_ndim_fac, &odd_buffer[0], 0, 0, 0);
        err = clEnqueueNDRangeKernel(queue, odd_tcf, 1, 0, &work_size, 0, 0, 0, 0);
        gpu_end_time = time(NULL);
        gpu_time += (gpu_end_time - gpu_start_time);

      } else {                
        cycle = is_even;
        for (I2 = 0; I2 < tcf_window * vec_ndim_fac; I2++) { 
          vector_file >> even_buffer[I2];
        }
        gpu_start_time = time(NULL);
        err = clEnqueueWriteBuffer(queue, cl_even_buffer, CL_TRUE, 0, sizeof(float) * tcf_window * vec_ndim_fac, &even_buffer[0], 0, 0, 0);
        err = clEnqueueNDRangeKernel(queue, even_tcf, 1, 0, &work_size, 0, 0, 0, 0); 
        gpu_end_time = time(NULL);
        gpu_time += (gpu_end_time - gpu_start_time);
      }
      std::cout << I1 << " gpu cycle completed\n";
    }
  
    if (tcf_last_window != 0) {
      err = clEnqueueWriteBuffer(queue, cl_tcf_window, CL_TRUE, 0, sizeof(int), &tcf_last_window, 0, 0, 0);
      std::cout << "Calculating last loop\n";
      size_t last_work_size = tcf_last_window;
      if (cycle == is_odd){
        for (I1 = 0; I1 < tcf_last_window * vec_ndim_fac; I1++) {
           vector_file >> even_buffer[I1];
        }
        gpu_start_time = time(NULL);
        err = clEnqueueWriteBuffer(queue, cl_even_buffer, CL_TRUE, 0, sizeof(float) * tcf_window * vec_ndim_fac, &even_buffer[0], 0, 0, 0);
        err = clEnqueueNDRangeKernel(queue, even_tcf, 1, 0, &last_work_size, 0, 0, 0, 0);
        gpu_end_time = time(NULL);
        gpu_time += (gpu_end_time - gpu_start_time);
      } else {
        for (I1 = 0; I1 < tcf_last_window * vec_ndim_fac; I1++) {
          vector_file >> odd_buffer[I1];
        }
        gpu_start_time = time(NULL);
        err = clEnqueueWriteBuffer(queue, cl_odd_buffer, CL_TRUE, 0, sizeof(float) * tcf_window * vec_ndim_fac, &odd_buffer[0], 0, 0, 0);
        err = clEnqueueNDRangeKernel(queue, odd_tcf, 1, 0, &last_work_size, 0, 0, 0, 0);
        gpu_end_time = time(NULL);
        gpu_time += (gpu_end_time - gpu_start_time);
      }
     err = clEnqueueWriteBuffer(queue, cl_tcf_window, CL_TRUE, 0, sizeof(int), &tcf_window, 0, 0, 0);
    }
    err = clEnqueueReadBuffer(queue, cl_tcf_buffer, CL_TRUE, 0, sizeof(float) * tcf_window, &tcf_buffer[0], 0, 0, 0);
    std::cout << "Output normalized TCF " << I0+1 << " to " << write_file_name << std::endl;
    for (I1= 0; I1 < tcf_window; I1++) {    
      tcf_file << tcf_buffer[I1]/tcf_buffer[0] << " ";
    }
    tcf_file << std::endl;
    for (I1 = 0; I1 < tcf_window; I1++) {
      tcf_buffer[I1] = 0;
    }
    err = clEnqueueWriteBuffer(queue, cl_tcf_buffer, CL_TRUE, 0, sizeof(float) * tcf_window, &tcf_buffer[0], 0, 0, 0);
  } 
  time_t end_time = time(NULL);
  std::cout << "Program finished! Cleaning memory!\n";
  clReleaseKernel(first_tcf);
  clReleaseKernel(odd_tcf);
  clReleaseKernel(even_tcf);
  clReleaseProgram(program);
  clReleaseMemObject(cl_ndim_fac);
  clReleaseMemObject(cl_tcf_window);
  clReleaseMemObject(cl_odd_buffer);
  clReleaseMemObject(cl_even_buffer);
  clReleaseMemObject(cl_tcf_buffer);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  delete [] odd_buffer;
  delete [] even_buffer;
  delete [] tcf_buffer;
  int running_hour = (end_time - start_time) / 3600;
  int running_min = ((end_time - start_time) % 3600) / 60;
  int running_sec = ((end_time - start_time) % 3600) % 60;
  int gpu_hour = ((int)(gpu_time)) / 3600;
  int gpu_min = (((int)(gpu_time)) % 3600) / 60;
  int gpu_sec = (((int)(gpu_time)) % 3600) % 60;
  std::cout << "Starting time = " << start_time << " , ending time = " << end_time << " (sec) since 1970/1/1 \n";
  std::cout << "The calculation of TCF costs " << running_hour << " hour(s) " << running_min << " min(s) " << running_sec << " sec(s) \n";
  std::cout << "Running_time/GPU_time (sec) = " << (end_time - start_time) << "/" << gpu_time << std::endl;
  std::cout << "Averaged gpu time per cycle for " << tcf_window << " time window " << " is " << gpu_time / (float)((gpu_cycle + last_gpu_cycle) * (tcf_num+1)) << " sec\n";
  tcf_file << "# Running_time/GPU_time (sec)= " << (end_time - start_time) << " / " << gpu_time << " , or running= " << running_hour << " h, " << running_min << " m, " << running_sec << "s vs. gpu= " << gpu_hour << " h, " << gpu_min << " m, " << gpu_sec << " s\n"; 
  vector_file.close();
  tcf_file.close();
  return 0;
}


