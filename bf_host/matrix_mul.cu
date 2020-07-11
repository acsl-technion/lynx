#include "bf_host.cu.hpp"


/***************** TEMPLATE *********************
__global__ void kernel_name(void **arg_buf, void **res_buf, unsigned int* recv_ci_addr, unsigned int* send_pi_addr, unsigned int* send_ci_addr) {
    __shared__ volatile  void *recv_buf;
    __shared__ volatile void *send_buf;
    if(threadIdx.x == 0) {
        recv_buf = arg_buf[getGlobalID()];
        send_buf = res_buf[getGlobalID()];
    }
	worker_init_send_recv(recv_buf, send_buf, recv_ci_addr, send_pi_addr, send_ci_addr);
	for(int i = 0 ; i < 100000*N ; i++) {
		grecv();
		do_something();
		gsend();
	}
}
**************************************************/


// A is NxN
#define A_N 256
#define A_M A_N
// A * b = scalar - A is a matrix stored as A transpose in global memory, b is a vector received from network
__global__ void matrix_mul(int* A, void **arg_buf, void **res_buf, unsigned int* recv_ci_addr, unsigned int* send_pi_addr, unsigned int* send_ci_addr) {
	__shared__ volatile  void *recv_buf;
	__shared__ volatile void *send_buf;
	if(threadIdx.x == 0) {
		recv_buf = arg_buf[getGlobalID()];
		send_buf = res_buf[getGlobalID()];
	}
    worker_init_send_recv(recv_buf, send_buf, recv_ci_addr, send_pi_addr, send_ci_addr);
	int tid = threadIdx.x;
	__shared__ int vector_b[A_N];
//    for(int i = 0 ; i < 100000*N ; i++) {
	volatile int kk = 1;
	while(kk) {
        grecv();
//		vector_b[tid] = *(int*)((int*)(recv_buf) + SOCKPERF_HEADER/4 + tid);
		copy_sockperf_header();
		__syncthreads();
//		printf("%d ", vector_b[tid]);
		//if A_M > A_N change result to result[A_M/A_N]
		int result = 0;
		for(int k = 0 ; k < A_N ; k++) {
//			for(int i = tid ; i < A_M ; i += blockDim.x) {
//				result += b_element * A[k * N + i];
//			}
			result += (vector_b[k] * A[k * A_N + tid]);
		}
		*(int*)(((int*)send_buf) + SOCKPERF_HEADER/4 + tid) = result;
		__syncthreads();
        gsend();
    }
}

void generate_A_transpose(int* A, int m, int n) {
	for(int i = 0 ; i < m ; i++) {
		for(int j = 0 ; j < n ; j++) {
			A[i * A_N + j] = i;
		}
	}	
}

void print_A(int* A, int m, int n) {
	std::cout << "A: " << std::endl;
	for(int i = 0 ; i < m ; i++) {
		for(int j = 0 ; j < n ; j++) {
			std::cout << A[j* A_N + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "A Transpose: " << std::endl;
	for(int i = 0 ; i < m ; i++) {
		for(int j = 0 ; j < n ; j++) {
			std::cout << A[i * A_N + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "-----------------------------------" << std::endl;
}	

int main(int argc, char *argv[]) {
	string interface = "enp1s0f0";
	unsigned int connections_num = 1;
	if(argc == 1) {
		std::cout << "No arguments were passed, use default values: interface: enp1s0f0, connections_num: 1" << std::endl;
	} else {
		if(argc > 3) {
			std::cerr << "Too many arguments: (" << argc - 1 << ") while expecting (2)." << std::endl;
			exit(1);
		}
		interface = argv[1];
		connections_num = atoi(argv[2]);
		std::cout << "Received argments: interface: " << interface << " connections_num: " << connections_num << std::endl;
	}

	hostContext host_ctx(interface, connections_num, TCP_PORT_NUM);
	void** d_req_base_addresses = host_ctx.getDeviceReqBaseAddresses();
    void** d_resp_base_addresses = host_ctx.getDeviceRespBaseAddresses();
	

	std::cout << "Launch Axb on gpu" << std::endl;
	int A[A_M * A_N];
	generate_A_transpose(A,A_M,A_N);
	print_A(A,A_M,A_N);
	int* d_A;
	CUDA_CHECK(cudaMalloc(&d_A, A_M * A_N * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_A, A, A_M * A_N * sizeof(int), cudaMemcpyHostToDevice));
	unsigned int threads_per_tb = A_N;
    matrix_mul<<<connections_num,threads_per_tb>>>(d_A ,d_req_base_addresses, d_resp_base_addresses, host_ctx.getRequestCIBaseAddress(), host_ctx.getResponsePIBaseAddress(),host_ctx.getResponseCIBaseAddress());

	std::cout << " -------------------------------------------------- start TEST -------------------------------------------------- " << std::endl;


	sleep(1000);
	
	host_ctx.waitDevice();
	
	return 0;
}

