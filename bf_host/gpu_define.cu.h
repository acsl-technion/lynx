#ifndef __GPU_DEFINE_H__
#define __GPU_DEFINE_H__

#include "../common/setup.hpp" 


//#define DEBUG_PRINTF( ... ) printf( __VA_ARGS__ )
#define DEBUG_PRINTF( ... )


#define FIRST_THREAD_IN_BLOCK() ((threadIdx.x + threadIdx.y + threadIdx.z) == 0)
#define BEGIN_SINGLE_THREAD __syncthreads(); if(FIRST_THREAD_IN_BLOCK()) { do {
#define END_SINGLE_THREAD } while(0); } __syncthreads();
#define getGlobalID() blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z

//assumption 1D threadBlocks
#define worker_init_send_recv(qp_recv_addr, qp_send_addr, recv_ci_addr, send_pi_addr, send_ci_addr) \
unsigned int worker_gid = getGlobalID();\
int stride = gridDim.x;\
volatile unsigned int *recv_ci_ptr = recv_ci_addr + worker_gid;\
volatile unsigned int *send_pi_ptr = send_pi_addr + worker_gid;\
volatile unsigned int *send_ci_ptr = send_ci_addr + worker_gid;\
volatile void* qp_recv_base_addr = qp_recv_addr;\
volatile void* qp_send_base_addr = qp_send_addr;\
if(threadIdx.x * blockDim.x + threadIdx.y == 0) send_buf = send_buf + worker_gid * HOST_SEND_MSG_SIZE;\


#define grecv() \
{\
BEGIN_SINGLE_THREAD\
	DEBUG_PRINTF("----- GRECV -----\n");\
	volatile unsigned int ci_val = MOD( *(recv_ci_ptr) + stride,HOST_MAX_RECV_WQES);\
	*(recv_ci_ptr) = ci_val;\
	*(int*)((char*)qp_recv_base_addr + (ci_val * (HOST_RECV_MSG_SIZE + OWNER_INT_SIZE)) + HOST_RECV_MSG_SIZE) = 0;\
    recv_buf = (char*)qp_recv_base_addr + (MOD(ci_val+stride,HOST_MAX_RECV_WQES) * (HOST_RECV_MSG_SIZE + OWNER_INT_SIZE));\
	volatile unsigned int* ptr = (unsigned int*) (recv_buf + HOST_RECV_MSG_SIZE);\
	while(*ptr == 0) {}\
	DEBUG_PRINTF("gid=%d after loop: ci_val=%d stride=%d\n",worker_gid, ci_val,stride);\
END_SINGLE_THREAD\
}



//do we need threadfence? before updating send_buf?
#define gsend() \
{\
BEGIN_SINGLE_THREAD\
	unsigned int ci_val = *send_ci_ptr;\
	unsigned int pi_val = *send_pi_ptr;\
	DEBUG_PRINTF("----- GSEND -----\n");\
    DEBUG_PRINTF("before loop: pi_val=%d ci_val=%d\n",pi_val, ci_val);\
	while(!CAN_PUSH(pi_val,ci_val,HOST_MAX_SEND_WQES)) {\
		ci_val = *send_ci_ptr;\
	}\
	pi_val = MOD(pi_val + stride,HOST_MAX_SEND_WQES);\
	*(send_pi_ptr) = pi_val;\
    DEBUG_PRINTF("after loop: new_pi_val=%d ci_val=%d\n",*(send_pi_ptr), ci_val);\
	send_buf = (char*)qp_send_base_addr + (MOD(pi_val + stride, HOST_MAX_SEND_WQES) * HOST_SEND_MSG_SIZE);\
END_SINGLE_THREAD\
}


#ifdef SOCKPERF
#define copy_sockperf_header()\
{\
	for(int i = threadIdx.x ; i < SOCKPERF_HEADER/4 ; i += blockDim.x) {\
		*(int*)((int*)send_buf + i) = *(int*)((int*)recv_buf + i);\
		if(i == 2) *(int*)((int*)send_buf + i) = *(int*)((int*)send_buf + i) & 0xFFFFFFFF00000000;\
	}\
}
#else
#define copy_sockperf_header()
#endif

#endif
