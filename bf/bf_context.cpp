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


#include "bf_context.hpp"


//#define __MEASURE_GPU_RTT__
#ifdef __MEASURE_GPU_RTT__
	#define G_N 100000
	double g_stats[G_N];
	int g_index = 0;
	bool g_start = false;
#endif

struct recv_wr_t {
    struct ibv_sge* recv_sg_list;
    struct ibv_send_wr* rdma_write_wr_list;
};

struct send_wr_t {
    ibv_sge* rdma_read_sg_list;
    ibv_send_wr* rdma_read_wr_list;

    ibv_sge* read_sg;
    ibv_send_wr* read_wr;

    ibv_sge* write_sg;
    ibv_send_wr* write_wr;
};

struct client_md {
	bool _valid;
    struct sockaddr_in _client_addr;
    socklen_t _client_addr_len;
#ifdef __MEASURE_GPU_RTT__
	double _time_stamp;
	char padding[64 - sizeof(bool) - sizeof(struct sockaddr_in) - sizeof(socklen_t) - sizeof(double)];
#else 
	char padding[64 - sizeof(bool) - sizeof(struct sockaddr_in) - sizeof(socklen_t)];
#endif
};


double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}

inline void BFContext::copy_data_to_host(ib_resources_t* host_ib_resources, unsigned int wr_id, ibv_send_wr* rdma_write_wr_list) {
	if (ibv_post_send(host_ib_resources->qp,&(rdma_write_wr_list[MOD(wr_id, BF_MAX_SEND_WQES)]), NULL)) {
		std::cerr << "ibv_post_send() failed (line" << __LINE__ << ")" << std::endl;
		exit(1);
	}

    int max_ncqes = 4;
    struct ibv_wc wc[max_ncqes];
    int ncqes = ibv_poll_cq(host_ib_resources->send_cq, max_ncqes, wc);
    if (ncqes < 0) {
        std::cerr << "ibv_poll_cq() failed" << std::endl;
        exit(1);
    }
}

int BFContext::poll_request_from_client(client_md* client_md_rbuf, ib_resources_t* client_ib_resources, unsigned int rbuf_index) {
	switch(_connection_type) {
		case UDP_CONNECTION:
			return poll_udp_request_from_client(client_md_rbuf, client_ib_resources, rbuf_index);
/*		case TCP_CONNECTION:
			return poll_tcp_request_from_client();
		case IB_CONNECTION:
			return poll_ib_request_from_client();*/
		default:
			std::cerr << "Unknown Connection Type:  " << _connection_type << std::endl;
			exit(1);
	}
	return -1;
}


inline void BFContext::notify_host(ib_resources_t* notify_ib_resources, ibv_send_wr* write_wr){
	struct ibv_send_wr *bad_wr;
	if (ibv_post_send(notify_ib_resources->qp, write_wr, &bad_wr)) {
		std::cerr << "ibv_post_send() failed (line " << __LINE__ << ")" << std::endl;
		exit(1);
	}
	notify_ib_resources->resp_sent = 0;
    notify_ib_resources->posted_wqes++;
	if(notify_ib_resources->update_wrap_around) {
		notify_ib_resources->update_wrap_around = false;
		notify_ib_resources->wrap_around = (notify_ib_resources->wrap_around + 1) % _workers_num;
	} else {
		notify_ib_resources->update_wrap_around = true;
	}
}


inline bool BFContext::pull_notification_from_host(ib_resources_t* notify_ib_resources, ibv_send_wr* read_wr, unsigned int *wr_id, unsigned int worker_id) {

	int pi_val = *(((int*)notify_ib_resources->lrecv_buf) + worker_id);
    int ci_val = *(((int*)notify_ib_resources->lsend_buf + _workers_num) + worker_id);
	//int ci_val = *wr_id;

	int available_slots = HAS_REQUEST(pi_val, ci_val, BF_MAX_RECV_WQES);
    available_slots = available_slots / _workers_num;
//	if(worker_id % ( (_workers_num-1)/2 + 1) != 0) {
//	if(worker_id == ( _workers_num - 1) || (worker_id != 0 && worker_id != 4 && worker_id != 8 && worker_id != 12) ){
//	if((_workers_num != 1 && worker_id == (_workers_num - 1)) || (worker_id != 0 && worker_id != (_workers_num/2 + 1)) ) {
//	if((worker_id != 0 && worker_id%8 == 0) || worker_id != _workers_num - 1) {
	if(available_slots > 0) {
    	*wr_id = MOD(ci_val + _workers_num, BF_MAX_SEND_WQES);
    }
	if(worker_id != _workers_num - 1) {		
//	if(worker_id != 0 && worker_id != _workers_num - 1) {
//	if(worker_id % 2 != 0) {	
		return available_slots > 0;
	}

//    std::cout << "worker_gid " << worker_id << " available slots " << available_slots << " pi_val " << pi_val << " ci_val " << ci_val << std::endl;

/*
    if(available_slots > 0){
		std::cout << "available_slots : " << available_slots << std::endl;
		std::cout << "pi = " << pi_val << " ci_val " << ci_val << std::endl;
	}

*/
	unsigned int* _load_factor = &notify_ib_resources->load_factor[worker_id];
//	if(*_load_factor != 0) std::cout << "load factor " << *_load_factor << std::endl;
	if( (2 * (*_load_factor)) < available_slots) {
		*_load_factor = available_slots / 2;
	}

	if(available_slots != *_load_factor) {
		return available_slots > 0;
	}

	*_load_factor = available_slots/2;
	
	if(notify_ib_resources->posted_wqes > 0) {
		int max_ncqes = 8;
		struct ibv_wc wc[max_ncqes];
	    int ncqes = ibv_poll_cq(notify_ib_resources->send_cq, max_ncqes, wc);
    	if (ncqes < 0) {
    		std::cerr << "ibv_poll_cq() failed" << std::endl;
	        exit(1);
    	}
		notify_ib_resources->posted_wqes -= ncqes;
	}
	if(notify_ib_resources->posted_wqes >= 5) { // the rest for posting rdma_write to update host_ci
		return available_slots > 0;
	}
	struct ibv_send_wr* bad_wr;
	if (ibv_post_send(notify_ib_resources->qp, read_wr, &bad_wr)) {
		std::cerr << "ibv_post_send() failed (line " << __LINE__ << ")" << std::endl;
		exit(1);
	}
	notify_ib_resources->posted_wqes++;
	return available_slots > 0;   
}

inline void BFContext::update_ci(ib_resources_t* notify_ib_resources, int worker_id) {
//	std::cout << "worker_id " << worker_id << " ci = " << *(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) << std::endl;
	*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) = MOD(*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) + _workers_num, BF_MAX_RECV_WQES);
//	*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num) = MOD(*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num) + requests_num, BF_MAX_RECV_WQES);
}

/*
inline void BFContext::push_ci_to_host() {
	ibv_send_wr* bad_wr;
	int max_ncqes = 8;
	ibv_wc wc[max_ncqes];
	int ncqes = 0;

    if (ibv_post_send(host_notify_ib_resources->qp, write_wr, &bad_wr)) {
        std::cerr << "ibv_post_send() failed (line " << __LINE__ << ")"  << std::endl;
        exit(1);
    }

    do {
        ncqes = ibv_poll_cq(host_notify_ib_resources->send_cq, max_ncqes, wc);
    } while (0);
    if (ncqes < 0) {
        std::cerr << "ibv_poll_cq() failed" << std::endl;
        exit(1);
    }
//    if (wc.status != IBV_WC_SUCCESS) {
//        std::cerr << "got CQE with error " << wc.status << " (line " << __LINE__ << ")" << std::endl;
//       exit(1);
//    }
//    assert(wc.opcode == IBV_WC_RDMA_WRITE);
	
}

*/
inline void BFContext::send_response(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr, unsigned int wr_id, unsigned int last_wr_id, unsigned int worker_id,bool post_rdma) {
	switch(_connection_type) {
		case UDP_CONNECTION:
			send_udp_response(client_md_rbuf, host_ib_resources, notify_ib_resources, client_ib_resources, send_wr, wr_id, last_wr_id, worker_id,post_rdma);
			break;
//		case TCP_CONNECTION:
//			send_tcp_response(wr_id);
//			break;
//		case IB_CONNECTION:
//			send_ib_response(wr_id);
//			break;
		defaulf:
			std::cerr << "Unknown Connection Type:  " << _connection_type << std::endl;
            exit(1);	
	}
}


inline unsigned int BFContext::get_worker_id_and_notify_host(ib_resources_t* notify_ib_resources, send_wr_t* send_wr, unsigned int rbuf_index) {
//    notify_ib_resources->resp_sent++;
	if(notify_ib_resources->resp_sent >= RECV_WQES_NUM/2) {
//      std::cout << "notify host resp_sent " << notify_ib_resources->resp_sent << std::endl;
		notify_host(notify_ib_resources, send_wr->write_wr);
	}
	notify_ib_resources->resp_sent++;
	int worker_id = (rbuf_index + notify_ib_resources->wrap_around * BF_MAX_RECV_WQES) % _workers_num;
	if( MOD(*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) + _workers_num , BF_MAX_RECV_WQES) != rbuf_index ) {
		unsigned int worker_id_1 = (rbuf_index + (notify_ib_resources->wrap_around == 0 ? _workers_num - 1 : notify_ib_resources->wrap_around - 1) * BF_MAX_RECV_WQES) % _workers_num;
		unsigned int worker_id_2 = (rbuf_index + (notify_ib_resources->wrap_around + 1) * BF_MAX_RECV_WQES) % _workers_num;
		if(MOD(*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id_1) + _workers_num , BF_MAX_RECV_WQES) == rbuf_index ) {
			worker_id = worker_id_1;
		} else {
			worker_id = worker_id_2;
		}
	}
//	std::cout << "wrap_around= " << notify_ib_resources->wrap_around << " rbuf_index " << rbuf_index << " worker_id = " << worker_id << " last ci " << *(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) << std::endl;
//	assert(MOD(*(((unsigned int*)(notify_ib_resources->lsend_buf)) + _workers_num + worker_id) + _workers_num , BF_MAX_RECV_WQES) == rbuf_index);
	return worker_id;	
}

inline void BFContext::send_udp_response(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr, unsigned int wr_id, unsigned int last_wr_id, unsigned int worker_id,bool post_rdma) {
	if(post_rdma) {
	    ibv_send_wr* bad_wr; 
		ibv_send_wr* rdma_read_wr_list = send_wr->rdma_read_wr_list;
	 
		unsigned int wrap_around = notify_ib_resources->wrap_around;
		if(wr_id != last_wr_id) {
			if (ibv_post_send(host_ib_resources->qp, &(rdma_read_wr_list[wr_id]), &bad_wr)) {
        		std::cerr << "ibv_post_send() failed (line " << __LINE__ << ")"  << std::endl;
		        exit(1);
    		}
	//		std::cout << "worker_gid " << worker_id << " requests msg id= " << wr_id << std::endl;
			//last_wr_id = wr_id;
		}
		return;
	}

    int max_ncqes = 16;
    struct ibv_wc wc[max_ncqes];
	int ncqes = ibv_poll_cq(host_ib_resources->send_cq, max_ncqes, wc);
    if (ncqes < 0) {
        std::cerr << "ibv_poll_cq() failed" << std::endl;
        exit(1);
    }

//	if(ncqes) std::cout << "ncqes= " << ncqes << std::endl;

    int requests_num = 0;
    for(int i = 0 ; i < ncqes ; i++) {
        assert(wc[i].status == IBV_WC_SUCCESS);
		if(wc[i].opcode == IBV_WC_RDMA_WRITE) {
        	continue;
        }

//          std::cerr << "got CQE with error " << wc[i].status << " (line " << __LINE__ << ")" << std::endl;
//          exit(1);
//      }
		assert(wc[i].opcode == IBV_WC_RDMA_READ);

//		std::cout << "sending msg wc[i].wr_id " << wc[i].wr_id << std::endl;
		int rbuf_index = wc[i].wr_id;
		char* send_buf = client_ib_resources->lsend_buf + rbuf_index * BF_H2C_MSG_SIZE;
//		std::cout << "rbuf_index " << rbuf_index << std::endl;
		assert(client_md_rbuf[rbuf_index]._valid == true);
//		if(rbuf_index == 1) std::cout << "rbuf_index to be released " << rbuf_index << "worker_id " << (rbuf_index + wrap_around * BF_MAX_RECV_WQES) % _workers_num << std::endl;
		int worker_id = get_worker_id_and_notify_host(notify_ib_resources, send_wr, rbuf_index);
		update_ci(notify_ib_resources,worker_id);
//		update_ci(notify_ib_resources,(rbuf_index + wrap_around * BF_MAX_RECV_WQES) % _workers_num);
  //      notify_ib_resources->resp_sent += requests_num;
		
#ifdef __MEASURE_GPU_RTT__		
		if(g_start) {
			g_stats[g_index++] = (get_time_msec() - client_md_rbuf[rbuf_index]._time_stamp) * 1000;
			if(g_index == G_N) {
				double min_val = 10000, max_val = 0, sum_val = 0;
				for(int i = 0 ; i < G_N ; i++) {
					sum_val += g_stats[i];
					min_val = min_val < g_stats[i] ? min_val : g_stats[i];
					max_val = max_val > g_stats[i] ? max_val : g_stats[i];
					std::cout << g_stats[i] <<" usec." << std::endl;
				}
				std::cout << "min: " << min_val << " usec." << std::endl;
				std::cout << "max: " << max_val << " usec." << std::endl;
				std::cout << "avg: " << sum_val / G_N << " usec." << std::endl;
			}
		} else {
			if(++g_index == G_N) {
				g_index = 0;
				g_start = true;
			}
		}
#endif
//		std::cout << "sending msg wc[i].wr_id " << wc[i].wr_id << std::endl;
/*		for(int i = 0 ; i < HOST_SEND_MSG_SIZE ; i++) {
			printf("%x ",send_buf[i]);
		} 
		printf("\n");*/
	    int ret=sendto(client_ib_resources->client_fd, send_buf, HOST_SEND_MSG_SIZE, MSG_CONFIRM, (struct sockaddr *)&(client_md_rbuf[rbuf_index]._client_addr), client_md_rbuf[rbuf_index]._client_addr_len);
    	if (ret < 0) {
        	perror("send udp response");
	        exit(1);
    	}
		client_md_rbuf[rbuf_index]._valid = false;
		requests_num++;
    }

	//printf("sent %d bytes\n", ret);
    //update ci in GPU
/*	if(requests_num > 0) {
//		update_ci(notify_ib_resources, requests_num);
		notify_ib_resources->resp_sent += requests_num;
	}*/
}


ib_resources_t* BFContext::setup_notifyQP_from_Host(ib_resources_t* client_ib_resources, int sfd) {
	ibv_context* context = client_ib_resources->context;
	ibv_pd* pd = client_ib_resources->pd;
	struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));

	struct ibv_mr *mr_recv;
	char *recv_buf = (char*) malloc(2 * _workers_num * sizeof(unsigned int));
	for(int i = 0 ; i < _workers_num ; i++) {
		*((unsigned int*)recv_buf + i) = BF_MAX_RECV_WQES - _workers_num + i;
	}
	for(int i = 0 ; i < _workers_num ; i++) {
        *((unsigned int*)recv_buf + _workers_num + i) = BF_MAX_RECV_WQES - _workers_num + i;
    }

	mr_recv = ibv_reg_mr(pd, recv_buf, 2 * _workers_num * sizeof(unsigned int), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	if (!mr_recv) {
		std::cerr << "ibv_reg_mr() failed for data_from_host" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_send;
	char *send_buf = (char*) malloc(2 * _workers_num * sizeof(unsigned int));
	for(int i = 0 ; i < _workers_num ; i++) {
		*((unsigned int*)send_buf + i) = BF_MAX_SEND_WQES - _workers_num + i;
	}
	for(int i = 0 ; i < _workers_num ; i++) {
        *((unsigned int*)send_buf + _workers_num + i) = BF_MAX_SEND_WQES - _workers_num + i;
    }
	mr_send = ibv_reg_mr(pd, send_buf, 2 * _workers_num * sizeof(unsigned int), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!mr_send) {	
		std::cerr << "ibv_reg_mr() failed for data_for_host" << std::endl;
		exit(1);
	}

	struct ibv_cq *recv_cq = ibv_create_cq(context, BF_RECV_CQ_SIZE, NULL, NULL, 0);
	if (!recv_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_cq *send_cq = ibv_create_cq(context, BF_SEND_CQ_SIZE, NULL, NULL, 0);
	if (!send_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.send_cq = send_cq;
	qp_init_attr.recv_cq = recv_cq;
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.cap.max_send_wr = NOTIFY_WQES_NUM;
	qp_init_attr.cap.max_recv_wr = 0;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 0;
    //qp_init_attr.cap.max_inline_data = 32;
	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
	if (!qp) {
		std::cerr << "ibv_create_qp() failed errno=" << errno << std::endl;
		exit(1);
	}

	struct ib_info_t server_info;
	int ret;
	ret = recv(sfd, &server_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "recv" << std::endl;
		exit(1);
	}
	ib_resources->rmr_recv_key = server_info.mkey_data_buffer;
	ib_resources->rmr_recv_addr = server_info.addr_data_buffer;
	ib_resources->rmr_send_key = server_info.mkey_response_buffer;
	ib_resources->rmr_send_addr = server_info.addr_response_buffer;

	struct ibv_port_attr port_attr;
	ret = ibv_query_port(context, PORT_NUM, &port_attr);
	if (ret) {
		std::cerr << "ibv_query_port() failed" << std::endl;
		exit(1);
	}
	struct ib_info_t my_info;
	my_info.lid = port_attr.lid;
	my_info.qpn = qp->qp_num;
	int gid_index = get_gid_index(context);
	if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
		std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
		exit(1);
	}

	ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		perror("send");
		std::cerr << "send" << std::endl;
		exit(1);
	}

	struct ibv_qp_attr qp_attr;
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_INIT;
	qp_attr.pkey_index = 0;
	qp_attr.port_num = PORT_NUM;
	qp_attr.qp_access_flags = 0;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTR;
	qp_attr.path_mtu = IBV_MTU_4096;
	qp_attr.dest_qp_num = server_info.qpn;
	qp_attr.rq_psn      = 0 ;
	qp_attr.max_dest_rd_atomic = 1;
	qp_attr.min_rnr_timer = 12;
	qp_attr.ah_attr.is_global = 1;
	qp_attr.ah_attr.grh.dgid = server_info.gid;
	qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
	qp_attr.ah_attr.grh.flow_label = 0;
	qp_attr.ah_attr.grh.hop_limit = 1;
	qp_attr.ah_attr.grh.traffic_class = 0;
	qp_attr.ah_attr.dlid = server_info.lid;
	qp_attr.ah_attr.sl = 0;
	qp_attr.ah_attr.src_path_bits = 0;
	qp_attr.ah_attr.port_num = PORT_NUM;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU| IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		std::cerr << "ibv_modify_qp() to RTR failed ret= " << ret << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn = 0;
	qp_attr.timeout = 14;
	qp_attr.retry_cnt = 7;
	qp_attr.rnr_retry = 7;
	qp_attr.max_rd_atomic = 1;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
	if (ret) {
		std::cerr << "ibv_modify_qp() to RTS failed" << std::endl;
		exit(1);
	}

	ib_resources->context = context;
	ib_resources->pd = pd;
	ib_resources->qp = qp;
	ib_resources->recv_cq = recv_cq;
	ib_resources->send_cq = send_cq;
	ib_resources->lrecv_buf = recv_buf;
	ib_resources->lmr_recv = mr_recv;
	ib_resources->lsend_buf = send_buf;
	ib_resources->lmr_send = mr_send;
	ib_resources->posted_wqes = 0;
	ib_resources->resp_sent = 0;
	ib_resources->wrap_around = 0;
	ib_resources->update_wrap_around = false;
		
//	ib_resources->load_factor = 0;
	ib_resources->load_factor = (unsigned int*) malloc(sizeof(unsigned int) * _workers_num);
	for(int i = 0 ; i < _workers_num ; i++) {
		ib_resources->load_factor[i] = 0;	
	}

	return ib_resources;
}


ib_resources_t* BFContext::setup_writeQP_to_Host(ib_resources_t* client_ib_resources, int sfd) {
    ibv_context* context = client_ib_resources->context;
    ibv_pd* pd = client_ib_resources->pd;

    struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));

    struct ibv_mr *mr_recv = client_ib_resources->lmr_recv;
    char *recv_buf = client_ib_resources->lrecv_buf;

    struct ibv_cq *send_cq = ibv_create_cq(context, BF_SEND_CQ_SIZE, NULL, NULL, 0);
    if (!send_cq) {
        std::cerr << "ERROR: ibv_create_cq() failed" << std::endl;
        exit(1);
    }

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = send_cq;
    qp_init_attr.recv_cq = send_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = BF_MAX_SEND_WQES;
    qp_init_attr.cap.max_recv_wr = 0;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 0;
//	qp_init_attr.cap.max_inline_data = 32;
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        std::cerr << "ibv_create_qp() failed errno=" << errno << std::endl;
        exit(1);
    }

    struct ibv_port_attr port_attr;
    int ret = ibv_query_port(context, PORT_NUM, &port_attr);
    if (ret) {
        std::cerr << "ibv_query_port() failed ret= " << ret << std::endl;
        exit(1);
    }

    struct ib_info_t my_info;
    my_info.lid = port_attr.lid;
    my_info.qpn = qp->qp_num;
    my_info.mkey_data_buffer = mr_recv->rkey;
    my_info.addr_data_buffer = (uintptr_t)mr_recv->addr;
    int gid_index = get_gid_index(context);
    if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
        std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
        exit(1);
    }
    ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("setup_writeQP_to_Host send");
        exit(1);
    }

    struct ib_info_t client_info;
    recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("setup_writeQP_to_Host recv");
        exit(1);
    }

    ib_resources->rmr_recv_key = client_info.mkey_data_buffer;
    ib_resources->rmr_recv_addr = client_info.addr_data_buffer;
    ib_resources->rmr_send_key = client_info.mkey_response_buffer;
    ib_resources->rmr_send_addr = client_info.addr_response_buffer;

    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = PORT_NUM;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; /* we'll allow client to RDMA write and read on this QP */
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
        exit(1);
    }

    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = client_info.qpn; /* qp number of client */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid = client_info.gid;
    qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.hop_limit = 1;
    qp_attr.ah_attr.grh.traffic_class = 0;
    qp_attr.ah_attr.dlid = client_info.lid; /* LID (L2 Address) of client */
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = PORT_NUM;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        std::cerr << "ibv_modify_qp() to RTR failed ret= " << ret << std::endl;
        exit(1);
    }
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        std::cerr << "bv_modify_qp() to RTS failed" << std::endl;
        exit(1);
    }

    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->qp = qp;
    ib_resources->recv_cq = send_cq;
    ib_resources->send_cq = send_cq;
    ib_resources->lrecv_buf = recv_buf;
    ib_resources->lmr_recv = mr_recv;

    return ib_resources;
}



ib_resources_t* BFContext::setup_readQP_from_Host(ib_resources_t* client_ib_resources, int sfd) {
	ibv_context* context = client_ib_resources->context;
	ibv_pd* pd = client_ib_resources->pd;

	struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));

	struct ibv_mr *mr_recv = client_ib_resources->lmr_send;
	char *recv_buf = client_ib_resources->lsend_buf;

	struct ibv_cq *send_cq = ibv_create_cq(context, BF_SEND_CQ_SIZE, NULL, NULL, 0);
	if (!send_cq) {
		std::cerr << "ERROR: ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.send_cq = send_cq;
	qp_init_attr.recv_cq = send_cq;
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.cap.max_send_wr = BF_MAX_SEND_WQES;
	qp_init_attr.cap.max_recv_wr = 0;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 0;
	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
	if (!qp) {
		std::cerr << "ibv_create_qp() failed errno=" << errno << std::endl;
		exit(1);
	}

	int ret;
	struct ibv_port_attr port_attr;
	ret = ibv_query_port(context, PORT_NUM, &port_attr);
	if (ret) {
		std::cerr << "ibv_query_port() failed ret= " << ret << std::endl;
		exit(1);
	}

	struct ib_info_t my_info;
	my_info.lid = port_attr.lid;
	my_info.qpn = qp->qp_num;
	my_info.mkey_data_buffer = mr_recv->rkey;
	my_info.addr_data_buffer = (uintptr_t)mr_recv->addr;
	int gid_index = get_gid_index(context);
	if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
		std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
		exit(1);
	}
	ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		perror("setup_dQP send");
		exit(1);
	}

	struct ib_info_t client_info;
	recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		perror("recv");
		exit(1);
	}

	ib_resources->rmr_recv_key = client_info.mkey_data_buffer;
	ib_resources->rmr_recv_addr = client_info.addr_data_buffer;
	ib_resources->rmr_send_key = client_info.mkey_response_buffer;
	ib_resources->rmr_send_addr = client_info.addr_response_buffer;

	struct ibv_qp_attr qp_attr;
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_INIT;
	qp_attr.pkey_index = 0;
	qp_attr.port_num = PORT_NUM;
	qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; // we'll allow client to RDMA write and read on this QP 
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTR;
	qp_attr.path_mtu = IBV_MTU_4096;
	qp_attr.dest_qp_num = client_info.qpn; // qp number of client 
	qp_attr.rq_psn      = 0 ;
	qp_attr.max_dest_rd_atomic = 1; // max in-flight RDMA reads 
	qp_attr.min_rnr_timer = 12;
	qp_attr.ah_attr.is_global = 1;
	qp_attr.ah_attr.grh.dgid = client_info.gid;
	qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
	qp_attr.ah_attr.grh.flow_label = 0;
	qp_attr.ah_attr.grh.hop_limit = 1;
	qp_attr.ah_attr.grh.traffic_class = 0;
	qp_attr.ah_attr.dlid = client_info.lid; // LID (L2 Address) of client
	qp_attr.ah_attr.sl = 0;
	qp_attr.ah_attr.src_path_bits = 0;
	qp_attr.ah_attr.port_num = PORT_NUM;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		std::cerr << "ibv_modify_qp() to RTR failed ret= " << ret << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn = 0;
	qp_attr.timeout = 14;
	qp_attr.retry_cnt = 7;
	qp_attr.rnr_retry = 7;
	qp_attr.max_rd_atomic = 1;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
	if (ret) {
		std::cerr << "bv_modify_qp() to RTS failed" << std::endl;
		exit(1);
	}

	ib_resources->context = context;
	ib_resources->pd = pd;
	ib_resources->qp = qp;
	ib_resources->recv_cq = send_cq;
	ib_resources->send_cq = send_cq;
	ib_resources->lrecv_buf = recv_buf;
	ib_resources->lmr_recv = mr_recv;

	return ib_resources;
}


ib_resources_t* BFContext::setup_connection_with_client(CONNECTION_TYPE connection_type, const string& interface, unsigned int port) {
	struct ib_resources_t* client_ib_resources;
	switch(connection_type) {
		case RECV_UDP_CONNECTION:
			client_ib_resources = setup_recv_udp_connection_with_client(interface, port);
			break;
		case SEND_UDP_CONNECTION:
            client_ib_resources = setup_send_udp_connection_with_client(interface, port);
            break;
/*		case TCP_CONNECTION: 
			client_ib_resources = setup_tcp_connection_with_client(interface);
			break;
		case IB_CONNECTION:
			client_ib_resources = setup_ib_connection_with_client(interface);
			break;*/
		default:
			std::cerr << "Unknown Connection Type:  " << connection_type << std::endl;
			exit(1);
	}
	return client_ib_resources;
}


ib_resources_t* BFContext::setup_recv_udp_connection_with_client(const string& interface, unsigned int udp_port) {
	struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
    int lfd;
    lfd = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(lfd, F_SETFL, O_NONBLOCK);
    if (lfd < 0) {
        std::cerr << "socket" << std::endl;
        exit(1);
    }
	
	struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(udp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        std::cerr << "bind" << std::endl;
        exit(1);
    }

    listen(lfd, 1);
	std::cout << "UDP Server is listening on port " << udp_port << std::endl;
	
	string device_name = ib_device_from_netdev(interface.c_str());
    struct ibv_context *context = ibv_open_device_by_name(device_name);

    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        std::cerr << "ibv_alloc_pd() failed" << std::endl;
        exit(1);
    }
	
	struct ibv_mr *mr_recv;
    char* recv_buf = (char*) malloc(BF_TOTAL_DATA_TO_HOST_SIZE);
    /*memset(recv_buf,0,BF_TOTAL_DATA_TO_HOST_SIZE);
    for(int i = 0 ; i < BF_MAX_RECV_WQES ; i++) {
        *(int*)(recv_buf + i * BF_C2H_MSG_SIZE + HOST_RECV_MSG_SIZE) = 1;
    }*/
    mr_recv = ibv_reg_mr(pd, recv_buf, BF_TOTAL_DATA_TO_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_recv) {
        std::cerr << "ibv_reg_mr() failed for recv_buf" << std::endl;
        exit(1);
    }

	ib_resources->client_fd = lfd;
    ib_resources->recv_buf_offset = 0;
    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->lrecv_buf = recv_buf;
    ib_resources->lmr_recv = mr_recv;

    return ib_resources;
}


	
ib_resources_t* BFContext::setup_send_udp_connection_with_client(const string& interface, unsigned int udp_port) {
    struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
    int lfd;
    lfd = socket(AF_INET, SOCK_DGRAM, 0);
    fcntl(lfd, F_SETFL, O_NONBLOCK);
    if (lfd < 0) {
        std::cerr << "socket" << std::endl;
        exit(1);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(udp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        std::cerr << "bind" << std::endl;
        exit(1);
    }

    listen(lfd, 1);
    std::cout << "UDP Server is listening on port " << udp_port << std::endl;

    string device_name = ib_device_from_netdev(interface.c_str());
    struct ibv_context *context = ibv_open_device_by_name(device_name);

    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        std::cerr << "ibv_alloc_pd() failed" << std::endl;
        exit(1);
    }

    struct ibv_mr *mr_send;
    char* send_buf = (char*) malloc(BF_TOTAL_DATA_FROM_HOST_SIZE);
    mr_send = ibv_reg_mr(pd, send_buf, BF_TOTAL_DATA_FROM_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!send_buf) {
        std::cerr << "ibv_reg_mr() failed for send_buf" << std::endl;
        exit(1);
    }

    ib_resources->client_fd = lfd;
    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->lsend_buf = send_buf;
    ib_resources->lmr_send = mr_send;

    return ib_resources;
}



/*
ib_resources_t* BFContext::setup_udp_connection_with_client(const string& interface, unsigned int udp_port) {

    struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
    int lfd, sfd;
    lfd = socket(AF_INET, SOCK_DGRAM, 0);
	fcntl(lfd, F_SETFL, O_NONBLOCK);
    if (lfd < 0) {
        std::cerr << "socket" << std::endl;
        exit(1);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(udp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        std::cerr << "bind" << std::endl;
        exit(1);
    }

    listen(lfd, 1);

    std::cout << "UDP Server is waiting on port " << udp_port << ". Client can connect" << std::endl;
	sfd = lfd;

    string device_name = ib_device_from_netdev(interface.c_str());
    struct ibv_context *context = ibv_open_device_by_name(device_name);

    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        std::cerr << "ibv_alloc_pd() failed" << std::endl;
        exit(1);
    }

    struct ibv_mr *mr_recv;
    char* recv_buf = (char*) malloc(BF_TOTAL_DATA_TO_HOST_SIZE);
    memset(recv_buf,0,BF_TOTAL_DATA_TO_HOST_SIZE);
    for(int i = 0 ; i < BF_MAX_RECV_WQES ; i++) {
        *(int*)(recv_buf + i * BF_C2H_MSG_SIZE + HOST_RECV_MSG_SIZE) = 1;
    }
	mr_recv = ibv_reg_mr(pd, recv_buf, BF_TOTAL_DATA_TO_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_recv) {
        std::cerr << "ibv_reg_mr() failed for recv_buf" << std::endl;
        exit(1);
    }


    struct ibv_mr *mr_send;
    char* send_buf = (char*) malloc(BF_TOTAL_DATA_FROM_HOST_SIZE);
    memset(recv_buf,0,BF_TOTAL_DATA_FROM_HOST_SIZE);
    mr_send = ibv_reg_mr(pd, send_buf, BF_TOTAL_DATA_FROM_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!send_buf) {
        std::cerr << "ibv_reg_mr() failed for send_buf" << std::endl;
        exit(1);
    }

    _client_sfd = sfd;
	_recv_buf_offset = 0;
	ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->lrecv_buf = recv_buf;
    ib_resources->lmr_recv = mr_recv;
    ib_resources->lsend_buf = send_buf;
    ib_resources->lmr_send = mr_send;

    return ib_resources;
}


ib_resources_t* BFContext::setup_tcp_connection_with_client(const string& interface) {

    struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
    int lfd, sfd;
    lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        std::cerr << "socket" << std::endl;
        exit(1);
    }

    int tcp_port = TCP_PORT_NUM;
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(tcp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        std::cerr << "bind" << std::endl;
        exit(1);
    }

    listen(lfd, 1);

    std::cout << "TCP Server is waiting on port " << tcp_port << ". Client can connect" << std::endl;
    sfd = accept(lfd, NULL, NULL);
    if (sfd < 0) {
        std::cerr << "accept" << std::endl;
        exit(1);
    }
    std::cout << "client is connected" << std::endl;

    string device_name = ib_device_from_netdev(interface.c_str());
    struct ibv_context *context = ibv_open_device_by_name(device_name);

    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        std::cerr << "ibv_alloc_pd() failed" << std::endl;
        exit(1);
    }

    struct ibv_mr *mr_recv;
    char* recv_buf = (char*) malloc(BF_TOTAL_DATA_TO_HOST_SIZE);
    mr_recv = ibv_reg_mr(pd, recv_buf, BF_TOTAL_DATA_TO_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_recv) {
        std::cerr << "ibv_reg_mr() failed for recv_buf" << std::endl;
        exit(1);
    }

    struct ibv_mr *mr_send;
    char* send_buf = (char*) malloc(BF_TOTAL_DATA_FROM_HOST_SIZE);
    mr_send = ibv_reg_mr(pd, send_buf, BF_TOTAL_DATA_FROM_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!send_buf) {
        std::cerr << "ibv_reg_mr() failed for send_buf" << std::endl;
        exit(1);
    }

	_client_sfd = sfd;
    _recv_buf_offset = 0;
    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->lrecv_buf = recv_buf;
    ib_resources->lmr_recv = mr_recv;
    ib_resources->lsend_buf = send_buf;
    ib_resources->lmr_send = mr_send;

    return ib_resources;
}


ib_resources_t* BFContext::setup_ib_connection_with_client(const string& interface) {
	
	struct ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
	int lfd, sfd;
	lfd = socket(AF_INET, SOCK_STREAM, 0);
	if (lfd < 0) {
		std::cerr << "socket" << std::endl;
		exit(1);
	}

	int tcp_port = TCP_PORT_NUM;
	struct sockaddr_in server_addr;
	memset(&server_addr, 0, sizeof(struct sockaddr_in));
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = INADDR_ANY;
	server_addr.sin_port = htons(tcp_port);

	if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
		std::cerr << "bind" << std::endl;
		exit(1);
	}

	listen(lfd, 1);

	std::cout << "IB Server is waiting on port " << tcp_port << ". Client can connect" << std::endl;
	sfd = accept(lfd, NULL, NULL);
	if (sfd < 0) {
		std::cerr << "accept" << std::endl;
		exit(1);
	}
	std::cout << "client is connected" << std::endl;

	string device_name = ib_device_from_netdev(interface.c_str());
	struct ibv_context *context = ibv_open_device_by_name(device_name);

	struct ibv_pd *pd = ibv_alloc_pd(context);
	if (!pd) {
		std::cerr << "ibv_alloc_pd() failed" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_recv;
	char* recv_buf = (char*) malloc(BF_TOTAL_DATA_TO_HOST_SIZE);
	memset(recv_buf,0,BF_TOTAL_DATA_TO_HOST_SIZE);
	for(int i = 0 ; i < BF_MAX_RECV_WQES ; i++) {
		*(int*)(recv_buf + i * BF_C2H_MSG_SIZE + BF_C2H_MSG_SIZE - sizeof(unsigned int)) = 1;	
	}
	mr_recv = ibv_reg_mr(pd, recv_buf, BF_TOTAL_DATA_TO_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	if (!mr_recv) {
		std::cerr << "ibv_reg_mr() failed for recv_buf" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_send;
	char* send_buf = (char*) malloc(BF_TOTAL_DATA_FROM_HOST_SIZE);
	mr_send = ibv_reg_mr(pd, send_buf, BF_TOTAL_DATA_FROM_HOST_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!send_buf) {
		std::cerr << "ibv_reg_mr() failed for send_buf" << std::endl;
		exit(1);
	}

	struct ibv_cq *recv_cq = ibv_create_cq(context, BF_RECV_CQ_SIZE, NULL, NULL, 0); // create a CQ with place for 100 CQEs
	if (!recv_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_cq *send_cq = ibv_create_cq(context, BF_SEND_CQ_SIZE, NULL, NULL, 0); // create a CQ with place for 100 CQEs
	if (!send_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.send_cq = send_cq;
	qp_init_attr.recv_cq = recv_cq;
	qp_init_attr.qp_type = IBV_QPT_RC; // we'll use RC transport service, which supports RDMA
	qp_init_attr.cap.max_send_wr = BF_MAX_SEND_WQES; // max of 1 WQE in-flight in SQ. that's enough for us
	qp_init_attr.cap.max_recv_wr = BF_MAX_RECV_WQES; // max of 8 WQE's in-flight in RQ. that's more than enough for us
	qp_init_attr.cap.max_send_sge = 1; // 1 SGE in each send WQE
	qp_init_attr.cap.max_recv_sge = 1; // 1 SGE in each recv WQE
	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
	if (!qp) {
		std::cerr << "ibv_create_qp() failed" << std::endl;
		exit(1);
	}

	int ret;
	struct ibv_port_attr port_attr;
	ret = ibv_query_port(context, PORT_NUM, &port_attr);
	if (ret) {
		std::cerr << "ERROR: ibv_query_port() failed ret= " << ret << std::endl;
		exit(1);
	}

	struct ib_info_t my_info;
	my_info.lid = port_attr.lid;
	my_info.qpn = qp->qp_num;
	my_info.mkey_data_buffer = mr_recv->rkey;
	my_info.addr_data_buffer = (uintptr_t)mr_recv->addr;
	my_info.mkey_response_buffer = mr_send->rkey;
	my_info.addr_response_buffer = (uintptr_t)mr_send->addr;
	int gid_index = get_gid_index(context);
	if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
		std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
		exit(1);
	}

	ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		perror("send");
		std::cerr << "send" << std::endl;
		exit(1);
	}

	struct ib_info_t client_info;
	ret = recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "recv" << std::endl;
		exit(1);
	}

	ib_resources->rmr_recv_key = client_info.mkey_data_buffer;
	ib_resources->rmr_recv_addr = client_info.addr_data_buffer;
	ib_resources->rmr_send_key = client_info.mkey_response_buffer;
	ib_resources->rmr_send_addr = client_info.addr_response_buffer;

	close(sfd);
	close(lfd);

	struct ibv_qp_attr qp_attr;
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_INIT;
	qp_attr.pkey_index = 0;
	qp_attr.port_num = PORT_NUM;
	qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; // we'll allow client to RDMA write and read on this QP
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTR;
	qp_attr.path_mtu = IBV_MTU_4096;
	qp_attr.dest_qp_num = client_info.qpn; // qp number of client
	qp_attr.rq_psn      = 0 ;
	qp_attr.max_dest_rd_atomic = 1; // max in-flight RDMA reads
	qp_attr.min_rnr_timer = 12;
	qp_attr.ah_attr.is_global = 1;
	qp_attr.ah_attr.grh.dgid = client_info.gid;
	qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
	qp_attr.ah_attr.grh.flow_label = 0;
	qp_attr.ah_attr.grh.hop_limit = 1;
	qp_attr.ah_attr.grh.traffic_class = 0;
	qp_attr.ah_attr.dlid = client_info.lid; // LID (L2 Address) of client
	qp_attr.ah_attr.sl = 0;
	qp_attr.ah_attr.src_path_bits = 0;
	qp_attr.ah_attr.port_num = PORT_NUM;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		std::cerr << "bv_modify_qp() to RTR failed ret= " <<  ret << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn = 0;
	qp_attr.timeout = 14;
	qp_attr.retry_cnt = 7;
	qp_attr.rnr_retry = 7;
	qp_attr.max_rd_atomic = 1;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
	if (ret) {
		std::cerr << "ibv_modify_qp() to RTS failed" << std::endl;
		exit(1);
	}

	ib_resources->context = context;
	ib_resources->pd = pd;
	ib_resources->qp = qp;
	ib_resources->recv_cq = recv_cq;
	ib_resources->send_cq = send_cq;
	ib_resources->lrecv_buf = recv_buf;
	ib_resources->lmr_recv = mr_recv;
	ib_resources->lsend_buf = send_buf;
	ib_resources->lmr_send = mr_send;

	return ib_resources;
}

*/
struct send_wr_t* BFContext::prepare_send_qps(ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources) {
    
    struct send_wr_t* send_wr = (struct send_wr_t*) malloc(sizeof(send_wr_t));

	send_wr->read_sg = (ibv_sge*)malloc(sizeof(ibv_sge));
	send_wr->read_wr = (ibv_send_wr*)malloc(sizeof(ibv_send_wr));
	memset(send_wr->read_sg, 0, sizeof(struct ibv_sge));
	memset(send_wr->read_wr, 0, sizeof(struct ibv_send_wr));

	send_wr->read_sg->addr = (uintptr_t)notify_ib_resources->lmr_recv->addr;
    send_wr->read_sg->length = 2 * _workers_num * sizeof(unsigned int);
    send_wr->read_sg->lkey = notify_ib_resources->lmr_recv->lkey;

	send_wr->read_wr->wr_id = 0; //This could be a problem when we need to debug
	send_wr->read_wr->sg_list = send_wr->read_sg;
	send_wr->read_wr->num_sge = 1;
	send_wr->read_wr->opcode = IBV_WR_RDMA_READ;
	send_wr->read_wr->send_flags = IBV_SEND_SIGNALED;
	send_wr->read_wr->wr.rdma.remote_addr = notify_ib_resources->rmr_send_addr;
	send_wr->read_wr->wr.rdma.rkey = notify_ib_resources->rmr_send_key;

	send_wr->write_sg = (ibv_sge*)malloc(sizeof(ibv_sge));
	send_wr->write_wr = (ibv_send_wr*)malloc(sizeof(ibv_send_wr));
	memset(send_wr->write_sg, 0, sizeof(struct ibv_sge));
    memset(send_wr->write_wr, 0, sizeof(struct ibv_send_wr));
    
	send_wr->write_sg->addr = (uintptr_t)notify_ib_resources->lmr_send->addr + _workers_num * sizeof(unsigned int);
	send_wr->write_sg->length = _workers_num * sizeof(unsigned int);
	send_wr->write_sg->lkey = notify_ib_resources->lmr_send->lkey;

	send_wr->write_wr->wr_id = 0; //This could be a problem when we need to debug
	send_wr->write_wr->sg_list = send_wr->write_sg;
	send_wr->write_wr->num_sge = 1;
	send_wr->write_wr->opcode = IBV_WR_RDMA_WRITE;
	send_wr->write_wr->send_flags = IBV_SEND_SIGNALED;
	send_wr->write_wr->wr.rdma.remote_addr = notify_ib_resources->rmr_recv_addr;
	send_wr->write_wr->wr.rdma.rkey = notify_ib_resources->rmr_recv_key;
    

	send_wr->rdma_read_sg_list = (ibv_sge*) malloc(BF_MAX_SEND_WQES * sizeof(ibv_sge));
	memset(send_wr->rdma_read_sg_list, 0, BF_MAX_SEND_WQES * sizeof(struct ibv_sge));
	send_wr->rdma_read_wr_list = (ibv_send_wr*)malloc(BF_MAX_SEND_WQES * sizeof(ibv_send_wr));
	memset(send_wr->rdma_read_wr_list, 0, BF_MAX_SEND_WQES * sizeof(struct ibv_send_wr));

    for(int i = 0 ; i < BF_MAX_SEND_WQES ; i++) {
        send_wr->rdma_read_sg_list[i].addr = (uintptr_t)client_ib_resources->lmr_send->addr + i * BF_H2C_MSG_SIZE;
        send_wr->rdma_read_sg_list[i].length = BF_H2C_MSG_SIZE;
        send_wr->rdma_read_sg_list[i].lkey = client_ib_resources->lmr_send->lkey;
    }

    for (int i = 0; i < BF_MAX_SEND_WQES; i++) {
        send_wr->rdma_read_wr_list[i].wr_id = i;
        send_wr->rdma_read_wr_list[i].sg_list = &(send_wr->rdma_read_sg_list[i]);
        send_wr->rdma_read_wr_list[i].num_sge = 1;
        send_wr->rdma_read_wr_list[i].opcode = IBV_WR_RDMA_READ;
        send_wr->rdma_read_wr_list[i].send_flags = IBV_SEND_SIGNALED;
        send_wr->rdma_read_wr_list[i].wr.rdma.remote_addr = host_ib_resources->rmr_send_addr + i * BF_H2C_MSG_SIZE;
        send_wr->rdma_read_wr_list[i].wr.rdma.rkey = host_ib_resources->rmr_send_key;
    }
	
	return send_wr;
}

struct recv_wr_t* BFContext::prepare_recv_qps(ib_resources_t* host_ib_resources, ib_resources_t* client_ib_resources) {
	struct recv_wr_t* recv_wr = (struct recv_wr_t*) malloc(sizeof(recv_wr_t));
	recv_wr->recv_sg_list = (ibv_sge*) malloc(BF_MAX_RECV_WQES * sizeof(ibv_sge));
	memset(recv_wr->recv_sg_list, 0, BF_MAX_RECV_WQES * sizeof(struct ibv_sge));

	for (int i = 0; i < BF_MAX_RECV_WQES; i++) {
		recv_wr->recv_sg_list[i].addr = (uintptr_t)client_ib_resources->lmr_recv->addr + i * BF_C2H_MSG_SIZE;
		recv_wr->recv_sg_list[i].length = BF_C2H_MSG_SIZE;
		recv_wr->recv_sg_list[i].lkey = client_ib_resources->lmr_recv->lkey;
	}

	recv_wr->rdma_write_wr_list = (ibv_send_wr*)malloc(BF_MAX_RECV_WQES * sizeof(ibv_send_wr));
	memset(recv_wr->rdma_write_wr_list, 0, BF_MAX_RECV_WQES * sizeof(struct ibv_send_wr));

	for (int i = 0; i < BF_MAX_RECV_WQES; i++) {
		recv_wr->rdma_write_wr_list[i].wr_id = i;
		recv_wr->rdma_write_wr_list[i].sg_list = &(recv_wr->recv_sg_list[i]); /* we could have used response_sg_list[i] */
		recv_wr->rdma_write_wr_list[i].num_sge = 1;
		recv_wr->rdma_write_wr_list[i].opcode = IBV_WR_RDMA_WRITE;
		recv_wr->rdma_write_wr_list[i].send_flags = IBV_SEND_SIGNALED;
		recv_wr->rdma_write_wr_list[i].wr.rdma.remote_addr = host_ib_resources->rmr_recv_addr + i * BF_C2H_MSG_SIZE;
		recv_wr->rdma_write_wr_list[i].wr.rdma.rkey = host_ib_resources->rmr_recv_key;
	}
	return recv_wr;
}

/*
int BFContext::get_host_sfd() {
	return _host_sfd;
}

int BFContext::get_client_sfd() {
    return _client_sfd;
}


ib_resources_t* BFContext::get_nQP_ib_resources() {
	return host_notify_ib_resources;
}


void BFContext::set_nQP_ib_resources(ib_resources_t* ib_resources) {
	host_notify_ib_resources = ib_resources;
}
*/

BFContext::BFContext(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, unsigned int workers_num) : _connection_type(connection_type), _host_port_num(host_port_num), _client_port_num(client_port_num), _workers_num(workers_num) { 
	switch(_connection_type) {
		case UDP_CONNECTION:
			_recv_connection_type = RECV_UDP_CONNECTION;
			_send_connection_type = SEND_UDP_CONNECTION;
			break;
		default:
			std::cout << "unknown connection type " << connection_type << std::endl;
	}
};


void BFContext::run_all() {
	client_md client_md_rbuf[BF_MAX_RECV_WQES];
	memset(client_md_rbuf, 0, BF_MAX_RECV_WQES * sizeof(client_md));
	
	thread tsend(&BFContext::send_thread, this, _connection_type, _host_port_num, _client_port_num + 50, client_md_rbuf);	
	recv_thread(_connection_type, _host_port_num, _client_port_num, client_md_rbuf);
	
	tsend.join();
}

BFContext::~BFContext() {}

/*
BFContext::BFContext(bool first_connection, bool last_connection, unsigned int connection_base_id, unsigned int connection_id, CONNECTION_TYPE connection_type, unsigned int connections_num, int sfd) : _first_connection(first_connection), _last_connection(last_connection), _connection_base_id(connection_base_id), _connection_id(connection_id), _connection_type(connection_type), _connections_num(connections_num) {
	
	std::cout << "Connection type: " << _connection_type << std::endl;
	std::cout << "wait for client to connect in order to create QPs between BlueField <==> Host" << std::endl;
	client_ib_resources = setup_connection_with_client("enp3s0f0", UDP_PORT_NUM + _connection_base_id + _connection_id);
	std::cout << "Client is connected" << std::endl;

	_last_wr_id = 1;
	_load_factor = 0;
	
	_host_sfd = sfd;
    if(_first_connection) {
		_host_sfd = socket(AF_INET, SOCK_STREAM, 0);
    	if (_host_sfd < 0) {
			std::cerr << "socket" << std::endl;
			exit(1);
		}
		struct sockaddr_in server_addr;
		server_addr.sin_addr.s_addr = inet_addr("192.168.0.20");
		server_addr.sin_family = AF_INET;
		server_addr.sin_port = htons(TCP_PORT_NUM + _connection_base_id + _connection_id);
	    if (connect(_host_sfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
    	    std::cerr << "connect" << std::endl;
        	exit(1);
	    }
	    std::cout << "BlueField is connected to Host" << std::endl;
	}

    if(_first_connection) {
        host_notify_ib_resources = setup_nQP_Host(_host_sfd);
        std::cout << "Host notify QP is established" << std::endl;
    } else {
        host_notify_ib_resources = NULL;
        std::cout << "Host notify QP is already established" << std::endl;
    }

	host_data_ib_resources = setup_dQP_Host(_host_sfd);
	std::cout << "Host data QP is established" << std::endl;


	if(_last_connection) {	//last one
		close(_host_sfd);
	}

	prepare_qps();

}


BFContext::~BFContext() {
	close(_client_sfd);
	ibv_destroy_qp(client_ib_resources->qp);
	ibv_destroy_qp(host_data_ib_resources->qp);
	if(host_notify_ib_resources != NULL) {
	    ibv_destroy_qp(host_notify_ib_resources->qp);
		ibv_destroy_cq(host_notify_ib_resources->recv_cq);
		ibv_destroy_cq(host_notify_ib_resources->send_cq);
		ibv_dereg_mr(host_notify_ib_resources->lmr_recv);
		ibv_dereg_mr(host_notify_ib_resources->lmr_send);
		free(host_notify_ib_resources->lrecv_buf);
		free(host_notify_ib_resources->lsend_buf);
		free(host_notify_ib_resources);	
		free(read_wr);
		free(read_sg);
		free(write_sg);
		free(write_wr);
	}
	ibv_destroy_cq(client_ib_resources->recv_cq);
	ibv_destroy_cq(host_data_ib_resources->recv_cq);
	ibv_destroy_cq(client_ib_resources->send_cq);
	ibv_destroy_cq(host_data_ib_resources->send_cq);
	ibv_dereg_mr(client_ib_resources->lmr_recv);
	ibv_dereg_mr(client_ib_resources->lmr_send);
	free(client_ib_resources->lrecv_buf);
	free(client_ib_resources->lsend_buf);
	ibv_dealloc_pd(client_ib_resources->pd);
	ibv_close_device(client_ib_resources->context);
	free(client_ib_resources);
	free(host_data_ib_resources);

	free(recv_sg_list);
	free(recv_wr_list);
	free(rdma_read_sg_list);
	free(rdma_read_wr_list);
	free(response_wr_list);
	free(rdma_write_wr_list);
}
*/


void BFContext::recv_thread(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, client_md* client_md_rbuf) {
	std::cout << "*** recv_thread ***" << std::endl;

	std::cout << "Client connection type: " << connection_type << std::endl;
    std::cout << "wait for client to connect in order to create QPs between BlueField <==> Host" << std::endl;
    ib_resources_t* client_ib_resources = setup_connection_with_client(_recv_connection_type, "enp3s0f0", client_port_num);

	std::cout << "Connect to Host" << std::endl;
	int host_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (host_fd < 0) {
		perror("socket");
		exit(1);
	}
	
	struct sockaddr_in server_addr;
	server_addr.sin_addr.s_addr = inet_addr("192.168.0.20");
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(host_port_num);
	if (connect(host_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
		perror("connect");
		exit(1);
	}
	std::cout << "BlueField is connected to Host,Data can be exchanged." << std::endl;

	ib_resources_t* host_ib_resources = setup_writeQP_to_Host(client_ib_resources, host_fd);
    std::cout << "Host write data QP is established" << std::endl;
	
	std::cout << "closing connection with Host" << std::endl;
	close(host_fd);

    recv_wr_t* recv_wr = prepare_recv_qps(host_ib_resources, client_ib_resources);

	recv_loop(client_md_rbuf, host_ib_resources, client_ib_resources, recv_wr->rdma_write_wr_list);
	//clean before exit
}


inline void BFContext::mark_owner_int(int* owner_int) {
     *owner_int = 1;
}


inline bool BFContext::can_push_to_host(client_md* client_md_rbuf, unsigned int rbuf_index) {
	bool ret = !client_md_rbuf[rbuf_index]._valid;
//	if(ret == false) std::cout << "recv_thread rbuf_index " << rbuf_index << std::endl;
	return ret;
}


inline int BFContext::recvfrom_client(client_md* client_md_rbuf, ib_resources_t* client_ib_resources, unsigned int rbuf_index) {
    char* recv_buf = client_ib_resources->lrecv_buf + client_ib_resources->recv_buf_offset * BF_C2H_MSG_SIZE;
    int ret = recvfrom(client_ib_resources->client_fd, recv_buf, HOST_RECV_MSG_SIZE, MSG_WAITALL, (struct sockaddr *)&(client_md_rbuf[rbuf_index]._client_addr), &(client_md_rbuf[rbuf_index]._client_addr_len));
    if(ret == -1) {
        return 0;
    }
/*	std::cout << "recevied packet rbuf_index " << std::endl;
	for(int i = 0 ; i < ret ; i++) {
		printf("%x ",recv_buf[i]);
	}
	printf("\n");*/
    assert(ret == HOST_RECV_MSG_SIZE);
//	std::cout << "recv packet" << std::endl;
#ifdef __MEASURE_GPU_RTT__
    client_md_rbuf[rbuf_index]._time_stamp = get_time_msec();
#endif
    client_md_rbuf[rbuf_index]._valid = true;
//	std::cout << "assign to rbuf_index " << rbuf_index << std::endl;
    mark_owner_int((int*)(recv_buf + ret));

    client_ib_resources->recv_buf_offset += 1;
    if(client_ib_resources->recv_buf_offset == BF_MAX_RECV_WQES) {
        client_ib_resources->recv_buf_offset = 0;
    }
    return 1;
}


int BFContext::poll_udp_request_from_client(client_md* client_md_rbuf, ib_resources_t* client_ib_resources, unsigned int rbuf_index) {
    if(!can_push_to_host(client_md_rbuf, rbuf_index)) {
        return 0;
    }

    return recvfrom_client(client_md_rbuf, client_ib_resources, rbuf_index);
}


static unsigned int  _worker_id = 0;
inline void BFContext::inc_worker_id() {
	_worker_id++;
    if(_worker_id == _workers_num){
        _worker_id = 0;
    }
}
inline unsigned int BFContext::get_free_worker() {
	return _worker_id;
}

void BFContext::recv_loop(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* client_ib_resources, ibv_send_wr* rdma_write_wr_list) {
	int rbuf_index[_workers_num];
	for(int i = 0 ; i < _workers_num ; i++) {
		rbuf_index[i] = i;
	}
	while(true) {
		unsigned int worker_id = get_free_worker();
		if(poll_request_from_client(client_md_rbuf, client_ib_resources, rbuf_index[worker_id]) == 0) {
			continue;
		}
        copy_data_to_host(host_ib_resources,rbuf_index[worker_id], rdma_write_wr_list);
		rbuf_index[worker_id] = rbuf_index[worker_id] + _workers_num;
		if(rbuf_index[worker_id] >= BF_MAX_RECV_WQES) {
			rbuf_index[worker_id] = rbuf_index[worker_id] % BF_MAX_RECV_WQES;
		}
		inc_worker_id();
	}
}


void BFContext::send_thread(CONNECTION_TYPE connection_type, unsigned int host_port_num, unsigned int client_port_num, client_md* client_md_rbuf) {
	sleep(1);
    std::cout << "*** send_thread ***" << std::endl;

    std::cout << "Client connection type: " << connection_type << std::endl;
    std::cout << "BF responses to clients on port " << client_port_num << std::endl;
    ib_resources_t* client_ib_resources = setup_connection_with_client(_send_connection_type, "enp3s0f0", client_port_num);

    std::cout << "Connect to Host" << std::endl;
	int host_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (host_fd < 0) {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_addr.s_addr = inet_addr("192.168.0.20");
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(host_port_num);
    if (connect(host_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("connect");
        exit(1);
    }
    std::cout << "BlueField is connected to Host,Data can be exchanged." << std::endl;

    ib_resources_t* host_ib_resources = setup_readQP_from_Host(client_ib_resources, host_fd);
    std::cout << "Host read data QP is established" << std::endl;
    ib_resources_t* notify_ib_resources = setup_notifyQP_from_Host(client_ib_resources, host_fd);
    std::cout << "Host notify QP is established" << std::endl;

    std::cout << "closing connection with Host" << std::endl;
    close(host_fd);

    send_wr_t* send_wr = prepare_send_qps(host_ib_resources, notify_ib_resources, client_ib_resources);

    send_loop(client_md_rbuf, host_ib_resources, notify_ib_resources, client_ib_resources, send_wr);
    //clean before exit
}


void BFContext::send_loop(client_md* client_md_rbuf, ib_resources_t* host_ib_resources, ib_resources_t* notify_ib_resources, ib_resources_t* client_ib_resources, send_wr_t* send_wr) {
	unsigned int wr_id[_workers_num];
	unsigned int last_wr_id[_workers_num];
	for(int i = 0 ; i < _workers_num ; i++) {
		wr_id[i] = BF_MAX_SEND_WQES - _workers_num + i;
		last_wr_id[i] = wr_id[i];
	}
    while(true) {
        double add_delay = get_time_msec(); //do not delete/ somehow it improves the performance!
		for(int i = 0 ; i < _workers_num ; i++) {
			bool should_send = pull_notification_from_host(notify_ib_resources, send_wr->read_wr, &wr_id[i], i);	
//			if(should_send) std::cout << "worker_id = " << i << " send id " << wr_id[i] << " wrap_around " << notify_ib_resources->wrap_around << std::endl;
			send_response(client_md_rbuf, host_ib_resources, notify_ib_resources, client_ib_resources, send_wr, wr_id[i], last_wr_id[i], i,true);
	        last_wr_id[i] = wr_id[i];
		}
		send_response(client_md_rbuf, host_ib_resources, notify_ib_resources, client_ib_resources, send_wr, 0, 0, 0, false);

/*	
//		std::cout << "resp_sent " << notify_ib_resources->resp_sent << std::endl;	
		if(notify_ib_resources->resp_sent >= RECV_WQES_NUM/2) {
//			std::cout << "notify host resp_sent " << notify_ib_resources->resp_sent << std::endl;		
			notify_host(notify_ib_resources, send_wr->write_wr);
    	}
*/
	}
}

