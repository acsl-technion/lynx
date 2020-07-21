/*
* Copyright (c) Maroun Tork, Lina Maudlej and Mark Silberstein
* All rights reserved.
* If used, please cite: PAPER NAME, AUTHORS, CONFERENCE WHERE PUBLISHED
*
* Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation and/or
* other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
* ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


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
