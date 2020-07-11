#include "../common/setup.hpp"
#include "bf_host.cu.hpp"
#include "gpu_define.cu.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/fcntl.h>

#define RECV_MSG_SIZE 800 //sockperf header 16 bytes + 28*28 image pixels
#define SEND_MSG_SIZE 20 //soxkpef header 16 bytes + 4 bytes for answer


__global__ void fuse_conv2d_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute) {
   float compute_local[2];
  __shared__ float pad_temp_shared[180];
  __shared__ float input1_shared[500];
  for (int yy_c_init = 0; yy_c_init < 2; ++yy_c_init) {
    compute_local[yy_c_init] = 0.000000e+00f;
  }
  if ((((int)threadIdx.z) * 9) < ((180 - ((int)threadIdx.y)) - ((int)threadIdx.x))) {
    pad_temp_shared[(((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 180) * 180) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) + ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) * 6))] = (((((1 <= (((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30)) && ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) < 29)) && ((1 - ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) <= (((int)blockIdx.x) * 2))) && ((((int)blockIdx.x) * 2) < (29 - ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)))) ? input0[(((((((int)blockIdx.x) * 2) + (((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 180) * 784)) + ((((((int)threadIdx.z) * 3) + ((int)threadIdx.y)) + ((int)threadIdx.x)) % 6)) + ((((((((int)threadIdx.z) * 9) + ((int)threadIdx.y)) + ((int)threadIdx.x)) / 6) % 30) * 28)) - 29)] : 0.000000e+00f);
  }
  if (((int)threadIdx.z) < (20 - (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25))) {
    input1_shared[((((((int)threadIdx.z) * 25) + ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25) * 25)) + (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 5) % 5) * 5))] = input1[((((((int)threadIdx.z) * 25) + ((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 25) * 25)) + (((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) / 5) % 5) * 5))];
  }
  __syncthreads();
  for (int ry_inner = 0; ry_inner < 5; ++ry_inner) {
    for (int rx_inner = 0; rx_inner < 5; ++rx_inner) {
      for (int yy_c = 0; yy_c < 2; ++yy_c) {
        compute_local[yy_c] = (compute_local[yy_c] + (pad_temp_shared[(((((((int)threadIdx.y) * 12) + ((int)threadIdx.x)) + (ry_inner * 6)) + rx_inner) + (yy_c * 6))] * input1_shared[(((((int)threadIdx.z) * 25) + (ry_inner * 5)) + rx_inner)]));
      }
    }
  }
  for (int yy_inner_inner_inner = 0; yy_inner_inner_inner < 2; ++yy_inner_inner_inner) {
    compute[(((((((int)blockIdx.x) * 2) + (((int)threadIdx.z) * 676)) + (((int)threadIdx.y) * 52)) + ((int)threadIdx.x)) + (yy_inner_inner_inner * 26))] = compute_local[yy_inner_inner_inner];
  }
}

__global__ void fuse_tanh_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (13520 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = tanhf(input0[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))]);
  }
}

__global__ void fuse_max_pool2d_kernel0( float* __restrict__ input0,  float* __restrict__ tensor) {
   float tensor_local[1];
  tensor_local[0] = -3.402823e+38f;
  for (int rv = 0; rv < 2; ++rv) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if ((((int)blockIdx.x) * 512) < (3380 - ((int)threadIdx.x))) {
        tensor_local[0] = max(tensor_local[0], input0[(((((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 3380) * 13520) + ((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 13) * 2)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 13) % 13) * 52)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 169) % 20) * 676)) + (rv * 26)) + rv1)]);
      }
    }
  }
  if ((((int)blockIdx.x) * 512) < (3380 - ((int)threadIdx.x))) {
    tensor[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 3380) * 3380) + (((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) % 13)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 13) % 13) * 13)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 169) % 20) * 169))] = tensor_local[0];
  }
}


__global__ void fuse_conv2d_1_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute) {
   float compute_local[1];
  __shared__ float pad_temp_shared[150];
  __shared__ float input1_shared[500];
  compute_local[0] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 10; ++rc_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner < 2; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) {
      if ((((int)threadIdx.z) * 3) < (30 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5))) {
        pad_temp_shared[(((((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 30) * 150) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) * 5)) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) + (((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 15) % 2) * 75))] = (((((1 <= (((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15)) && ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) < 14)) && ((1 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) <= ((int)blockIdx.x))) && (((int)blockIdx.x) < (14 - (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)))) ? input0[((((((((int)blockIdx.x) + (rc_outer * 338)) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 30) * 3380)) + ((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) % 15) * 13)) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) % 5)) + (((((((int)threadIdx.z) * 3) + (((((int)threadIdx.y) * 2) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner) / 5)) / 15) % 2) * 169)) - 14)] : 0.000000e+00f);
      }
    }
    for (int ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1 < 5; ++ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1) {
      if (((int)threadIdx.z) < (10 - (((int)threadIdx.y) / 10))) {
        if ((((int)blockIdx.z) * 10) < ((50 - ((int)threadIdx.z)) - (((int)threadIdx.y) / 10))) {
          input1_shared[(((((((int)threadIdx.z) * 50) + ((((int)threadIdx.y) / 10) * 50)) + ((((int)threadIdx.y) % 5) * 5)) + (((((int)threadIdx.y) / 5) % 2) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)] = input1[(((((((((int)blockIdx.z) * 5000) + (rc_outer * 50)) + (((int)threadIdx.z) * 500)) + ((((int)threadIdx.y) / 10) * 500)) + ((((int)threadIdx.y) % 5) * 5)) + (((((int)threadIdx.y) / 5) % 2) * 25)) + ax0_ax1_fused_ax2_fused_ax3_fused_inner_inner_inner1)];
        }
      }
    }
    __syncthreads();
for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
      for (int ry_inner = 0; ry_inner < 5; ++ry_inner) {
        for (int rx_inner = 0; rx_inner < 5; ++rx_inner) {
          compute_local[0] = (compute_local[0] + (pad_temp_shared[((((((int)threadIdx.y) * 5) + (rc_inner * 75)) + (ry_inner * 5)) + rx_inner)] * input1_shared[((((((int)threadIdx.z) * 50) + (rc_inner * 25)) + (ry_inner * 5)) + rx_inner)]));
        }
      }
    }
  }
  compute[((((((int)blockIdx.z) * 1210) + ((int)blockIdx.x)) + (((int)threadIdx.z) * 121)) + (((int)threadIdx.y) * 11))] = compute_local[0];
}

__global__ void fuse_tanh_1_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (6050 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = tanhf(input0[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))]);
  }
}

__global__ void fuse_max_pool2d_1_kernel0( float* __restrict__ input0,  float* __restrict__ tensor) {
   float tensor_local[1];
  tensor_local[0] = -3.402823e+38f;
  for (int rv = 0; rv < 2; ++rv) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
        tensor_local[0] = max(tensor_local[0], input0[(((((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 6050) + ((((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 5) * 2)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 5) % 5) * 22)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 25) % 50) * 121)) + (rv * 11)) + rv1)]);
      }
    }
  }
  if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
    tensor[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 1250) + (((((int)blockIdx.x) * 2) + ((int)threadIdx.x)) % 5)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 5) % 5) * 5)) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 25) % 50) * 25))] = tensor_local[0];
  }
}

__global__ void fuse_flatten_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  if ((((int)blockIdx.x) * 512) < (1250 - ((int)threadIdx.x))) {
    tensor[((((int)blockIdx.x) * 512) + ((int)threadIdx.x))] = input0[(((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) / 1250) * 1250) + (((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) / 25) * 25)) + ((((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) / 5) % 5) * 5)) + ((((((int)blockIdx.x) * 512) + ((int)threadIdx.x)) % 1250) % 5))];
  }
}

__global__ void fuse_dense_kernel0( float* __restrict__ input0,  float* __restrict__ input1,  float* __restrict__ compute,  float* __restrict__ input2) {
   float compute_rf[1];
  __shared__ float red_buf0[64];
  __shared__ float compute1[1];
  compute_rf[0] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 20; ++k_outer) {
    if (((int)threadIdx.x) < (1250 - (k_outer * 64))) {
      compute_rf[0] = (compute_rf[0] + (input0[(((int)threadIdx.x) + (k_outer * 64))] * input1[(((((int)blockIdx.x) * 1250) + ((int)threadIdx.x)) + (k_outer * 64))]));
    }
  }
  ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = compute_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(32 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(16 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(8 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(4 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(2 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(1 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    compute1[0] = ((volatile __shared__ float*)red_buf0)[0];
  }
 if (((int)threadIdx.x) == 0) {
    compute[((int)blockIdx.x)] = (compute1[0] + input2[((int)blockIdx.x)]);
  }
}

__global__ void fuse_softmax_kernel0( float* __restrict__ tensor,  float* __restrict__ input0) {
  tensor[0] = -3.402823e+38f;
  for (int k1 = 0; k1 < 10; ++k1) {
    tensor[0] = max(tensor[0], input0[k1]);
  }
}

__global__ void fuse_softmax_kernel1( float* __restrict__ input0,  float* __restrict__ tensor,  float* __restrict__ tensor1) {
   float tensor_rf[1];
  __shared__ float red_buf0[64];
  tensor_rf[0] = 0.000000e+00f;
  if (((int)threadIdx.x) < 10) {
    tensor_rf[0] = (tensor_rf[0] + __expf((input0[((int)threadIdx.x)] - tensor[0])));
  }
  ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = tensor_rf[0];
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(32 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(16 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(8 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(4 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(2 + ((int)threadIdx.x))]);
    ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(1 + ((int)threadIdx.x))]);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tensor1[0] = ((volatile __shared__ float*)red_buf0)[0];
  }
}
__global__ void fuse_softmax_kernel2( float* __restrict__ tensor,  float* __restrict__ input0,  float* __restrict__ tensor1,  float* __restrict__ tensor2) {
  if (((int)threadIdx.x) < 10) {
    tensor[((int)threadIdx.x)] = (__expf((input0[((int)threadIdx.x)] - tensor1[0])) / tensor2[0]);
//	printf("out: %f\n", tensor[((int)threadIdx.x)]);
  }
}


/*__device__ volatile void *recv_buf;
__device__ volatile void *send_buf;*/

__global__ void convf(volatile char *in_data, float *data, volatile float* oo) {
	if(threadIdx.x < 4) {
		*((volatile unsigned int *)oo + threadIdx.x) = *((volatile unsigned int *)in_data+threadIdx.x);
		if(threadIdx.x == 2)
				*(volatile int *)&oo[threadIdx.x] = 0x00000000;//*(int *)&oo[threadIdx.x] & 0xFFFFFF00;
//		if(threadIdx.x == 3)
//				*(volatile int *)&oo[threadIdx.x] = 0x00001400;//*(int *)&oo[threadIdx.x] & 0xFFFFFF00;
		
	} else {

	    int x;
    	float o;
		int index = 16 + threadIdx.x - 4;
	    x = (volatile unsigned char)in_data[index] - 127;
    	data[threadIdx.x - 4] = __fdividef((float)x, 128);
	//    printf("%f\n",data[threadIdx.x - 4]);
	}
}

__global__ void lenet(float *data,
        float *conv1_weight,
        float *conv2_weight,
        float *fc2_weight,
        float *fc2_bias,
        float *o0,
        float *o1,
        float *o2,
        float *o3,
        float *o4,
		void **arg_buf, 
		void **res_buf, 
		unsigned int* recv_ci_addr, 
		unsigned int* send_pi_addr, 
		unsigned int* send_ci_addr
) {
	volatile void *recv_buf;
    volatile void *send_buf;
    recv_buf = arg_buf[0];
    send_buf = res_buf[0];
    worker_init_send_recv(recv_buf, send_buf, recv_ci_addr, send_pi_addr, send_ci_addr);
    int tid = threadIdx.x;

    dim3 dimGrid_convf(1,1,1),  dimBlock_convf(788,1,1);
    dim3 dimGrid_conv2d(13,1,1),  dimBlock_conv2d(2,13,20);
    dim3 dimGrid_tanh(27,1,1),  dimBlock_tanh(512,1,1);
    dim3 dimGrid_pool2d(7,1,1),  dimBlock_pool2d(512,1,1);
    dim3 dimGrid_conv2d1(11,1,5),  dimBlock_conv2d1(1,11,10);
    dim3 dimGrid_tanh1(12,1,1),  dimBlock_tanh1(512,1,1);
    dim3 dimGrid_pool2d1(3,1,1),  dimBlock_pool2d1(512,1,1);
    dim3 dimGrid_flatten(3,1,1),  dimBlock_flatten(512,1,1);
    dim3 dimGrid_dense(10,1,1),  dimBlock_dense(64,1,1);
    dim3 dimGrid_softmax0(1,1,1),  dimBlock_softmax0(1,1,1);
    dim3 dimGrid_softmax1(1,1,1),  dimBlock_softmax1(64,1,1);
    dim3 dimGrid_softmax2(1,1,1),  dimBlock_softmax2(64,1,1);
 
for(int ii= 0 ; ii < 10000000 ; ii++) {
//for(int ii= 0 ; ii < 1000 ; ii++) {
	grecv();
//    long long int start = clock64();	
	convf<<<dimGrid_convf,dimBlock_convf>>>((volatile char *)recv_buf, data, (volatile float*)send_buf);
    cudaDeviceSynchronize();
    fuse_conv2d_kernel0<<<dimGrid_conv2d,dimBlock_conv2d>>>(data, conv1_weight, o1);
    cudaDeviceSynchronize();
    fuse_tanh_kernel0<<<dimGrid_tanh,dimBlock_tanh>>>(o2,o1);
    cudaDeviceSynchronize();
    fuse_max_pool2d_kernel0<<<dimGrid_pool2d,dimBlock_pool2d>>>(o2, o1);
    cudaDeviceSynchronize();
    fuse_conv2d_1_kernel0<<<dimGrid_conv2d1,dimBlock_conv2d1>>>(o1, conv2_weight, o2);
    cudaDeviceSynchronize();
    fuse_tanh_1_kernel0<<<dimGrid_tanh1,dimBlock_tanh1>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_max_pool2d_1_kernel0<<<dimGrid_pool2d1,dimBlock_pool2d1>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_flatten_kernel0<<<dimGrid_flatten,dimBlock_flatten>>>(o1, o2);
    cudaDeviceSynchronize();
    fuse_dense_kernel0<<<dimGrid_dense,dimBlock_dense>>>(o1, fc2_weight, o3, fc2_bias);
    cudaDeviceSynchronize();
    fuse_softmax_kernel0<<<dimGrid_softmax0,dimBlock_softmax0>>>(o4, o3);
    cudaDeviceSynchronize();
    fuse_softmax_kernel1<<<dimGrid_softmax1,dimBlock_softmax1>>>(o3, o4, o0);
    cudaDeviceSynchronize();
    fuse_softmax_kernel2<<<dimGrid_softmax2,dimBlock_softmax2>>>( ((float*)send_buf+4), o3, o4, o0);
  //  long long int end = clock64();
	//printf("time in GPU %d usec\n",(end-start)/875);
/*	for(int i = 0 ; i < 10 ; i++) {
		printf("%d\n",((float*)send_buf+4+i));
	}*/
	gsend();
}
}



void load_input_from_file(const char* fname, char* buffer, unsigned int size, unsigned int offset) {
    int fd, ret;
    fd = open(fname, O_RDONLY);
    float *h_buff = new float[size];
    ret = pread(fd, h_buff, size, offset);
    if (ret != size) {
        perror("read error");
        delete[] h_buff;
        exit(-1);
    }
    unsigned char h_input[784];

    for (int i = 0; i < 784; i++) {
        h_input[i] = (h_buff[i] *128) + 127;
    }

    cudaMemcpy(buffer, h_input, 784, cudaMemcpyHostToDevice);
    delete[] h_buff;
    close(fd);
}

void load_from_file(const char* fname, float* buffer, unsigned int size, unsigned int offset) {
    int fd, ret;
    fd = open(fname, O_RDONLY);
    float *h_buff = new float[size];
    ret = pread(fd, h_buff, size, offset);
    if (ret != size) {
        perror("read error");
        delete[] h_buff;
        exit(-1);
    }
    cudaMemcpy(buffer, h_buff, size, cudaMemcpyHostToDevice);
    delete[] h_buff;
    close(fd);
}



int main(int argc, char *argv[]) {
	string interface = "enp1s0f0";
	unsigned int total_workers_num = 1;
    unsigned int cores_num=1;
    if(argc == 1) {
        std::cout << "No arguments were passed, use default values: interface: enp1s0f0, total_workers_num: 1, cores_num: 1" << std::endl;
    } else {
        if(argc > 4) {
            std::cerr << "Too many arguments: (" << argc - 1 << ") while expecting (4)." << std::endl;
            exit(1);
        }
        interface = argv[1];
        total_workers_num = atoi(argv[2]);
        cores_num = atoi(argv[3]);
        std::cout << "Received argments: interface: " << interface << " total_workers_num: " << total_workers_num << std::endl;
    }

    cudaStream_t streams[cores_num];
    std::vector<hostContext*> host_ctx;
    std::vector<void**> d_req_base_addresses;
    std::vector<void**> d_resp_base_addresses;

	int gpus_num = 2;

    float *o0[gpus_num], *o1[gpus_num], *o2[gpus_num], *o3[gpus_num], *o4[gpus_num], *oo[gpus_num], *data[gpus_num], *conv1_weight[gpus_num], *conv2_weight[gpus_num], *fc2_weight[gpus_num], *fc2_bias[gpus_num];
    char *indata[gpus_num];
    float output[gpus_num][14];

	for(int i = 0 ; i < gpus_num ; i++) {
			cudaSetDevice(i);
			cudaMalloc(&o0[i], sizeof(float)* 1024);
			cudaMalloc(&o1[i], sizeof(float)* 13520);
			cudaMalloc(&o2[i], sizeof(float)* 13520);
			cudaMalloc(&o3[i], sizeof(float)* 10);
			cudaMalloc(&o4[i], sizeof(float)* 1024);
			cudaMalloc(&oo[i], sizeof(float)* 10 + 16);
			cudaMalloc(&data[i], sizeof(float)* 28 * 28);
			cudaMalloc(&conv1_weight[i], sizeof(float)* 500);
			cudaMalloc(&conv2_weight[i], sizeof(float)* 25000);
			cudaMalloc(&fc2_weight[i], sizeof(float)* 12500);
			cudaMalloc(&fc2_bias[i], sizeof(float)* 10);

			load_from_file("data2.dat", conv1_weight[i],2000,0);
			load_from_file("data0.dat", conv2_weight[i],100000,0);
			load_from_file("data1.dat", fc2_weight[i],50000,0);
			load_from_file("data3.dat", fc2_bias[i],40,0);
	}


    for(int i = 0 ; i < cores_num ; i++) {
		cudaSetDevice(i);
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        unsigned int tbs_num = total_workers_num / cores_num + (total_workers_num % cores_num > i ? 1 : 0 );
        host_ctx.push_back(new hostContext(interface, tbs_num, TCP_PORT_NUM + i));

        d_req_base_addresses.push_back(host_ctx[i]->getDeviceReqBaseAddresses());
        d_resp_base_addresses.push_back(host_ctx[i]->getDeviceRespBaseAddresses());

    }

    for(int i = 0 ; i < cores_num ; i++) {
		cudaSetDevice(i);
    	lenet<<<1,1>>>(data[i], conv1_weight[i], conv2_weight[i], fc2_weight[i], fc2_bias[i], o0[i], o1[i], o2[i], o3[i], o4[i], d_req_base_addresses[i], d_resp_base_addresses[i], host_ctx[i]->getRequestCIBaseAddress(), host_ctx[i]->getResponsePIBaseAddress(),host_ctx[i]->getResponseCIBaseAddress());
	}

    std::cout << " -------------------------------------------------- start TEST -------------------------------------------------- " << std::endl;


//	CUDA_CHECK(cudaDeviceSynchronize());
//	std::cout << "done" << std::endl;
    sleep(1000);

	return 0;
}
