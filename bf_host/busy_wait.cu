#include "bf_host.cu.hpp"

#include <vector>

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


__global__ void busy_wait(void **arg_buf, void **res_buf, unsigned int* recv_ci_addr, unsigned int* send_pi_addr, unsigned int* send_ci_addr, unsigned int* useconds) {
    __shared__ volatile void *recv_buf;
    __shared__ volatile void *send_buf;
    if(threadIdx.x == 0) {
        recv_buf = arg_buf[0];
        send_buf = res_buf[0];
    }
    worker_init_send_recv(recv_buf, send_buf, recv_ci_addr, send_pi_addr, send_ci_addr);

    unsigned int usec = *useconds;
    long long int cycles_per_usec = 875;
    long long int curr;

    int kk = 1;
    while(kk > 0) {
		grecv();
        copy_sockperf_header();
        for(int i = threadIdx.x + SOCKPERF_HEADER/4 ; i < HOST_RECV_MSG_SIZE/4 ; i += blockDim.x ) {
            *(int*)((int*)send_buf + i) = *(int*)((int*)recv_buf + i);
        }
        __syncthreads();
        long long int start = clock64();
        long long int total_cycles = usec * cycles_per_usec;
        while(total_cycles >  0) {
            curr = clock64();
            total_cycles -= (curr - start);
            start = curr;
        }
        gsend();
    }
}



int main(int argc, char *argv[]) {
	string interface = "enp1s0f0";
	unsigned int total_workers_num = 1;
	unsigned int cores_num=1;
    unsigned int useconds = 0;
	if(argc == 1) {
		std::cout << "No arguments were passed, use default values: interface: enp1s0f0, total_workers_num: 1, cores_num: 1, delay in usec: 0" << std::endl;
	} else {
		if(argc != 5) {
			std::cerr << "Too few/many arguments: (" << argc - 1 << ") while expecting (4)." << std::endl;
			exit(1);
		}
		interface = argv[1];
		total_workers_num = atoi(argv[2]);
		cores_num = atoi(argv[3]);
		useconds = atoi(argv[4]);
		std::cout << "Received argments: interface: " << interface << " total_workers_num: " << total_workers_num << " cores number " << cores_num << " usecs delay " << useconds << std::endl;
	}

	cudaStream_t streams[cores_num];
	std::vector<hostContext*> host_ctx;
	std::vector<void**> d_req_base_addresses;
	std::vector<void**> d_resp_base_addresses;

	
	unsigned int* d_useconds;
    CUDA_CHECK(cudaMalloc(&d_useconds, sizeof(unsigned int)));
	CUDA_CHECK(cudaMemcpy(d_useconds, &useconds, sizeof(unsigned int), cudaMemcpyHostToDevice));

		
	for(int i = 0 ; i < cores_num ; i++) {
		CUDA_CHECK(cudaStreamCreate(&streams[i]));	
        unsigned int tbs_num = total_workers_num / cores_num + (total_workers_num % cores_num > i ? 1 : 0);
		host_ctx.push_back(new hostContext(interface, tbs_num, TCP_PORT_NUM + i));

        d_req_base_addresses.push_back(host_ctx[i]->getDeviceReqBaseAddresses());
        d_resp_base_addresses.push_back(host_ctx[i]->getDeviceRespBaseAddresses());

	}	
	
	for(int i = 0 ; i < cores_num ; i++) {
		unsigned int threads_per_tb = 1;
        unsigned int tbs_num = total_workers_num / cores_num + (i == 0 ? total_workers_num % cores_num : 0 );
		busy_wait<<<tbs_num,threads_per_tb,0, streams[i]>>>(d_req_base_addresses[i], d_resp_base_addresses[i], host_ctx[i]->getRequestCIBaseAddress(), host_ctx[i]->getResponsePIBaseAddress(),host_ctx[i]->getResponseCIBaseAddress(),d_useconds);
	}
	
	std::cout << " -------------------------------------------------- start TEST -------------------------------------------------- " << std::endl;


	sleep(1000);

//	CUDA_CHECK(cudaDeviceSynchronize());	
	//host_ctx.waitDevice();
	
	return 0;
}

