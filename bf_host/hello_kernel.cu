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


__global__ void hello_kernel(void **arg_buf, void **res_buf, unsigned int* recv_ci_addr, unsigned int* send_pi_addr, unsigned int* send_ci_addr) {
    __shared__ volatile  void *recv_buf;
    __shared__ volatile void *send_buf;
    if(threadIdx.x == 0) {
        recv_buf = arg_buf[getGlobalID()];
        send_buf = res_buf[getGlobalID()];
    }
	//printf("arg_buf=0x%p\n",arg_buf);

	worker_init_send_recv(recv_buf, send_buf, recv_ci_addr, send_pi_addr, send_ci_addr);
	for(int i = 0 ; i < 100000*N ; i++) {
		grecv();
		//printf("recv_buf=0x%p val=%c\n",recv_buf,*(char*)recv_buf);
	/*	while(*(char*)recv_buf != 'T'){
			__threadfence();
		}*/
		for(int k = 0 ; k < 26 ; k++) {
			*((char*)send_buf + k) = (char) (i%26 + (getGlobalID() ? 'A' : 'a'));
		}
		*((char*)send_buf + 26) = '\0';
		gsend();
	}
}


int main(int argc, char *argv[]) {
	string interface = "enp130s0f0";
	unsigned int connections_num = 1;
	if(argc == 1) {
		std::cout << "No arguments were passed, use default values: interface: enp130s0f0, connections_num: 1" << std::endl;
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
	
	std::cout << "Launch hello kernel on gpu" << std::endl;
	unsigned int threads_per_tb = 1;
	hello_kernel<<<connections_num,threads_per_tb>>>(d_req_base_addresses, d_resp_base_addresses, host_ctx.getRequestCIBaseAddress(), host_ctx.getResponsePIBaseAddress(),host_ctx.getResponseCIBaseAddress());

	std::cout << " -------------------------------------------------- start TEST -------------------------------------------------- " << std::endl;


	sleep(1000);
	
	host_ctx.waitDevice();
	
	return 0;
}

