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


#ifndef __BF_CONTEXT_H__
#define __BF_CONTEXT_H__

#include "../common/setup.hpp"

#include <thread>
using namespace std;


typedef enum { 
	UDP_CONNECTION = 0,
	RECV_UDP_CONNECTION = 1,
	SEND_UDP_CONNECTION = 2,
	TCP_CONNECTION = 3,
	IB_CONNECTION = 4
} CONNECTION_TYPE;

struct recv_wr_t;
struct send_wr_t;
struct client_md;

class BFContext {
/*	int _host_sfd;
	bool _first_connection;
	bool _last_connection;
	unsigned int _connection_base_id;
	unsigned int _connection_id;*/
	unsigned int _workers_num;
	CONNECTION_TYPE _connection_type;
	CONNECTION_TYPE _recv_connection_type;
	CONNECTION_TYPE _send_connection_type;
	unsigned int _host_port_num;
	unsigned int _client_port_num;
/*
	unsigned int _last_wr_id;
	unsigned int _load_factor;
	
	// FOR TCP/UDP connections
	int _client_sfd;
	struct sockaddr_in _client_addr;
    socklen_t _client_addr_len;
	unsigned int _recv_buf_offset;

    ib_resources_t* client_ib_resources;
    ib_resources_t* host_notify_ib_resources;
    ib_resources_t* host_data_ib_resources;

    unsigned int polled_wqes;
    unsigned int next_client_wqes_batch_id;
    ibv_recv_wr* recv_wr_list_batch[BF_NUM_OF_WQE_LISTS];
    ibv_recv_wr* recv_wr_list;
    ibv_sge* recv_sg_list;
    ibv_sge* read_sg;
    ibv_send_wr* read_wr;
    ibv_sge* write_sg;
    ibv_send_wr* write_wr;
    ibv_sge* rdma_read_sg_list;
    ibv_send_wr* rdma_read_wr_list;
    ibv_send_wr* rdma_write_wr_list;
    ibv_send_wr* response_wr_list;
*/
	ib_resources_t* setup_connection_with_client(CONNECTION_TYPE connection_type, const string& interface, unsigned int port);
//    ib_resources_t* setup_ib_connection_with_client(const string& interface);
    ib_resources_t* setup_recv_udp_connection_with_client(const string& interface, unsigned int udp_port);
	ib_resources_t* setup_send_udp_connection_with_client(const string& interface, unsigned int udp_port);
	//ib_resources_t* setup_udp_connection_with_client(const string&, unsigned int);
//    ib_resources_t* setup_tcp_connection_with_client(const string& interface);

    int poll_request_from_client(client_md* client_md_rbuf, ib_resources_t* ib_resources, unsigned int rbuf_index = 0);
//	int poll_ib_request_from_client();
	int poll_udp_request_from_client(client_md* client_md_rbuf, ib_resources_t* ib_resources, unsigned int rbuf_index);
//	int poll_tcp_request_from_client();

	inline void send_response(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr, unsigned int wr_id, unsigned int last_wr_id, unsigned int worker_id,bool post_rdma);
	inline void send_udp_response(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr, unsigned int wr_id, unsigned int last_wr_id, unsigned int worker_id,bool post_rdma);
//	void send_ib_response(unsigned int wr_id);
//	void send_tcp_response(unsigned int wr_id);

	ib_resources_t* setup_writeQP_to_Host(ib_resources_t* client_ib_resources, int sfd);
	ib_resources_t* setup_notifyQP_from_Host(ib_resources_t* client_ib_resources, int sfd);
	ib_resources_t* setup_readQP_from_Host(ib_resources_t* client_ib_resources, int sfd);

	struct recv_wr_t* prepare_recv_qps(ib_resources_t* host_ib_resources, ib_resources_t* client_ib_resources);
	struct send_wr_t* prepare_send_qps(ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources);

//	void post_client_wqes_batch();
	void copy_data_to_host(ib_resources_t* host_ib_resources, unsigned int wr_id, ibv_send_wr* rdma_write_wr_list);
//	inline int get_free_slots_num();
	inline void notify_host(ib_resources_t* notify_ib_resources, ibv_send_wr* write_wr);
	inline bool pull_notification_from_host(ib_resources_t* notify_ib_resources, ibv_send_wr* read_wr, unsigned int *wr_id, unsigned int worker_id);
	inline void update_ci(ib_resources_t* notify_ib_resources, int requests_num);
//	inline void push_ci_to_host();
//	inline void update_ci(int requests_num);

	void recv_loop(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* client_ib_resources, ibv_send_wr* rdma_write_wr_list);
	void recv_thread(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, client_md* client_md_rbuf);
	void send_loop(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr);
	void send_thread(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, client_md* client_md_rbuf);

	inline void mark_owner_int(int* owner_int);
	inline bool can_push_to_host(client_md* client_md_rbuf, unsigned int rbuf_index);
	inline int recvfrom_client(client_md* client_md_rbuf, ib_resources_t* ib_resources, unsigned int rbuf_index);

	inline unsigned int get_worker_id_and_notify_host(ib_resources_t* notify_ib_resources, send_wr_t* send_wr, unsigned int rbuf_index);
	inline unsigned int get_free_worker();
	inline void inc_worker_id();

public:

	BFContext(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, unsigned int workers_num = 1);
	~BFContext();
	void run_all();	
//	void set_nQP_ib_resources(ib_resources_t* ib_resources);
//	ib_resources_t* get_nQP_ib_resources();
//	int get_host_sfd();
//	int get_client_sfd();
	
//	void recv_and_send_udp();

};

#endif
