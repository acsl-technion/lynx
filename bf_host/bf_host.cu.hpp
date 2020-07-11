#ifndef __BF_HOST_H__
#define __BF_HOST_H__

#include "../common/setup.hpp"
#include "gpu_define.cu.h"

class hostContext {
	unsigned int _workers_num;
    void** _d_req_base_addresses;
    void** _d_resp_base_addresses;

    ib_resources_t* recv_data_ib_resources;
	ib_resources_t* send_data_ib_resources;
    ib_resources_t* notify_ib_resources;

	void teardown_connection(ib_resources_t* ib_resources);
    ib_resources_t* setup_recv_data_connection(const string& interface, int sfd);
	ib_resources_t* setup_send_data_connection(const string& interface, int sfd);
	ib_resources_t* setup_notify_connection(const string& interface, int sfd);

public:
    hostContext(const string& interface, unsigned int workers_num = 1, unsigned int tcp_port = TCP_PORT_NUM);
	~hostContext();

	void* getRequestBaseAddress();
	void* getResponseBaseAddress();
	unsigned int* getRequestCIBaseAddress();
	unsigned int* getResponsePIBaseAddress();
	unsigned int* getResponseCIBaseAddress();
	void** getDeviceReqBaseAddresses();
	void** getDeviceRespBaseAddresses();
	void waitDevice();
};


#endif
