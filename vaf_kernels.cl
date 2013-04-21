__kernel void first_tcf(__global const int* ndim_fac, __global int* tcf_window, __global float* odd_buffer, __global float* tcf_buffer)
{
  int I0, J0, work_id = get_global_id(0);
  for (I0 = 0; I0 < *tcf_window - work_id; I0++) {   //self correlation
    for (J0 = 0; J0 < *ndim_fac; J0++) {
      tcf_buffer[work_id] += odd_buffer[I0 * *ndim_fac+ J0] * odd_buffer[(I0 + work_id) * *ndim_fac + J0];    
    }
  }
}

__kernel void odd_tcf(__global const int* ndim_fac, __global int* tcf_window, __global float* odd_buffer, __global float* even_buffer, __global float* tcf_buffer)
{
  int I0, J0, work_id = get_global_id(0);
  for (I0 = 0; I0 < *tcf_window - work_id; I0++) {   //self correlation
    for (J0 = 0; J0 < *ndim_fac; J0++) {
      tcf_buffer[work_id] += odd_buffer[I0 * *ndim_fac + J0] * odd_buffer[(I0 + work_id) * *ndim_fac + J0];
    }
  }
  for (I0 = 0; I0 < work_id; I0++) {               //cross correlation
    for (J0 = 0; J0 < *ndim_fac; J0++) { 
      tcf_buffer[work_id] += odd_buffer[I0 * *ndim_fac + J0] * even_buffer[(I0 + *tcf_window - work_id) * *ndim_fac + J0];
    }
  }
}

__kernel void even_tcf(__global const int* ndim_fac, __global int* tcf_window, __global float* odd_buffer, __global float* even_buffer, __global float* tcf_buffer)
{
  int I0, J0, work_id = get_global_id(0);
  for (I0 = 0; I0 < *tcf_window - work_id; I0++) {   //self correlation
    for (J0 = 0; J0 < *ndim_fac; J0++) {
      tcf_buffer[work_id] += even_buffer[I0 * *ndim_fac + J0] * even_buffer[(I0 + work_id) * *ndim_fac + J0];
    }
  }
  for (I0 = 0; I0 < work_id; I0++) {           //cross correlation
    for (J0 = 0; J0 < *ndim_fac; J0++) {
      tcf_buffer[work_id] += even_buffer[I0 * *ndim_fac + J0] * odd_buffer[(I0 + *tcf_window - work_id) * *ndim_fac + J0];
    }
  }
}


