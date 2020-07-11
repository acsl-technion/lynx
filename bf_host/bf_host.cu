#include "bf_host.cu.hpp"

void hostContext::teardown_connection(ib_resources_t* ib_resources) {
        ibv_destroy_qp(ib_resources->qp);
        ibv_destroy_cq(ib_resources->recv_cq);
        ibv_destroy_cq(ib_resources->send_cq);
        ibv_dereg_mr(ib_resources->lmr_recv);
        ibv_dereg_mr(ib_resources->lmr_send);
        free(ib_resources->lrecv_buf);
        free(ib_resources->lsend_buf);
        ibv_dealloc_pd(ib_resources->pd);
        ibv_close_device(ib_resources->context);
        free(ib_resources);
}


ib_resources_t* hostContext::setup_notify_connection(const string& interface, int sfd) {
	ib_resources_t *ib_resources = (struct ib_resources_t *)malloc(sizeof(struct ib_resources_t));
    struct ibv_device **device_list = ibv_get_device_list(NULL);
	if (!device_list) {
		std::cerr << "ibv_get_device_list failed" << std::endl;
		exit(1);
	}

	string device_name = ib_device_from_netdev(interface.c_str());
	struct ibv_context *context = ibv_open_device_by_name(device_name);

	struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
		std::cerr << "ibv_alloc_pd() failed" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_recv;
	char* recv_buf;
	CUDA_CHECK(cudaMalloc(&recv_buf, _workers_num * sizeof(unsigned int)));
	unsigned int recv_arr_size =  _workers_num;
	unsigned int recv_init[recv_arr_size];
	for(int i = 0 ; i < recv_arr_size ; i++) {
		recv_init[i] = HOST_MAX_RECV_WQES - _workers_num + i;
	}
	CUDA_CHECK(cudaMemcpy(recv_buf, recv_init, _workers_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
	mr_recv = ibv_reg_mr(pd, recv_buf, _workers_num * sizeof(unsigned int), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
	if (!mr_recv) {
		std::cerr << "ibv_reg_mr() failed for data_for_host" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_send;
	char *send_buf;
	CUDA_CHECK(cudaMalloc(&send_buf,2 * _workers_num * sizeof(unsigned int)));
	unsigned int send_arr_size = 2 * _workers_num;
	unsigned int send_init[send_arr_size];
	for(int i = 0 ; i < _workers_num ; i++) {
		send_init[i] = HOST_MAX_SEND_WQES - _workers_num + i;
	}
    for(int i = 0 ; i < _workers_num ; i++) {
        send_init[_workers_num + i] = HOST_MAX_SEND_WQES - 2 * _workers_num + i; //will be inc. when calling grecv
    }

/*	for(int i = 0 ; i < send_arr_size ; i++) {
        if( i < send_arr_size/2 ) { // PI part
            send_init[i] = HOST_MAX_SEND_WQES - 1;//0;
        } else { // CI part
            send_init[i] = HOST_MAX_SEND_WQES - 2; // will be inc. when calling grecv
        }
    }*/
	CUDA_CHECK(cudaMemcpy(send_buf, send_init, 2 * _workers_num * sizeof(unsigned int), cudaMemcpyHostToDevice));
	mr_send = ibv_reg_mr(pd, send_buf, 2 * _workers_num * sizeof(unsigned int), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!mr_send) {
		std::cerr << "ibv_reg_mr() failed for data_from_host" << std::endl;
		exit(1);
	}

	struct ibv_cq *recv_cq = ibv_create_cq(context, HOST_RECV_CQ_SIZE, NULL, NULL, 0);
	if (!recv_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_cq *send_cq = ibv_create_cq(context, HOST_SEND_CQ_SIZE, NULL, NULL, 0);
	if (!send_cq) {
		std::cerr << "ibv_create_cq() failed" << std::endl;
		exit(1);
	}

	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.send_cq = send_cq;
	qp_init_attr.recv_cq = recv_cq;
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.cap.max_send_wr = 0;
	qp_init_attr.cap.max_recv_wr = HOST_MAX_RECV_WQES;
	qp_init_attr.cap.max_send_sge = 0;
	qp_init_attr.cap.max_recv_sge = 1;
//	qp_init_attr.cap.max_inline_data = 512;
	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
	if (!qp) {
		std::cerr << "ibv_create_qp() failed errno= " << errno << std::endl;
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
	my_info.mkey_response_buffer = mr_send->rkey;
	my_info.addr_response_buffer = (uintptr_t)mr_send->addr;
	int gid_index = get_gid_index(context);
	if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
		std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
		exit(1);
	}
	ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "send" << std::endl;
		exit(1);
	}
	
	struct ib_info_t client_info;
	recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "recv" << std::endl;
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
	qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
		exit(1);
	}

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTR;
	qp_attr.path_mtu = IBV_MTU_4096;
	qp_attr.dest_qp_num = client_info.qpn;
	qp_attr.rq_psn      = 0 ;
	qp_attr.max_dest_rd_atomic = 1;
	qp_attr.min_rnr_timer = 12;
	qp_attr.ah_attr.is_global = 1;
	qp_attr.ah_attr.grh.dgid = client_info.gid;
	qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
	qp_attr.ah_attr.grh.flow_label = 0;
	qp_attr.ah_attr.grh.hop_limit = 1;
	qp_attr.ah_attr.grh.traffic_class = 0;
	qp_attr.ah_attr.dlid = client_info.lid;
	qp_attr.ah_attr.sl = 0;
	qp_attr.ah_attr.src_path_bits = 0;
	qp_attr.ah_attr.port_num = PORT_NUM;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		std::cerr << "ibv_modify_qp() to RTR failed ret= " << ret<< std::endl;
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


ib_resources_t* hostContext::setup_recv_data_connection(const string& interface, int sfd) {
    ib_resources_t *ib_resources = (ib_resources_t *)malloc(sizeof(struct ib_resources_t));

    ibv_device **device_list = ibv_get_device_list(NULL);
    if (!device_list) {
        std::cerr << "ERROR: ibv_get_device_list failed" << std::endl;
        exit(1);
    }

    string device_name = ib_device_from_netdev(interface.c_str());
    struct ibv_context *context = ibv_open_device_by_name(device_name);
    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        std::cerr << "ibv_alloc_pd() failed" << std::endl;
        exit(1);
    }
	
	struct ibv_mr *mr_recv;
    char *recv_buf;
    CUDA_CHECK(cudaMalloc(&recv_buf,HOST_TOTAL_DATA_FROM_CLIENT_SIZE));
    CUDA_CHECK(cudaMemset(recv_buf, 0, HOST_TOTAL_DATA_FROM_CLIENT_SIZE));
//    printf("ib_resources Data: recv_buf=%p size=%d\n",recv_buf,HOST_TOTAL_DATA_FROM_CLIENT_SIZE);
    mr_recv = ibv_reg_mr(pd, recv_buf, HOST_TOTAL_DATA_FROM_CLIENT_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_recv) {
        std::cerr << "ibv_reg_mr() failed for data_for_host" << std::endl;
        exit(1);
    }

    struct ibv_cq *recv_cq = ibv_create_cq(context, HOST_RECV_CQ_SIZE, NULL, NULL, 0);
    if (!recv_cq) {
        printf("ERROR: ibv_create_cq() failed\n");
        exit(1);
    }

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = recv_cq;
    qp_init_attr.recv_cq = recv_cq;
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = 0;
    qp_init_attr.cap.max_recv_wr = HOST_MAX_RECV_WQES;
    qp_init_attr.cap.max_send_sge = 0;
    qp_init_attr.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        std::cerr << "ibv_create_qp() failed errno= " << errno << std::endl;
        exit(1);
    }

    struct ibv_port_attr port_attr;
    int ret = ibv_query_port(context, PORT_NUM, &port_attr);
    if (ret) {
        std::cerr << "ibv_query_port() failed ret=" << ret << std::endl;
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
        std::cerr << "send" << std::endl;
        exit(1);
    }

    struct ib_info_t client_info;
    recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        std::cerr << "recv" << std::endl;
        exit(1);
    }

    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = PORT_NUM;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
        exit(1);
    }

    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = client_info.qpn;
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid = client_info.gid;
    qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.hop_limit = 1;
    qp_attr.ah_attr.grh.traffic_class = 0;
    qp_attr.ah_attr.dlid = client_info.lid;
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
        std::cerr << "ibv_modify_qp() to RTS failed" << std::endl;
        exit(1);
    }

    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->qp = qp;
    ib_resources->recv_cq = recv_cq;
    ib_resources->lrecv_buf = recv_buf;
    ib_resources->lmr_recv = mr_recv;

    return ib_resources;
}



ib_resources_t* hostContext::setup_send_data_connection(const string& interface, int sfd) {
	ib_resources_t *ib_resources = (ib_resources_t *)malloc(sizeof(struct ib_resources_t));

	ibv_device **device_list = ibv_get_device_list(NULL);
	if (!device_list) {
		std::cerr << "ERROR: ibv_get_device_list failed" << std::endl;
		exit(1);
	}

	string device_name = ib_device_from_netdev(interface.c_str());
	struct ibv_context *context = ibv_open_device_by_name(device_name);
	struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
		std::cerr << "ibv_alloc_pd() failed" << std::endl;
		exit(1);
	}

	struct ibv_mr *mr_send;
	char *send_buf;
	CUDA_CHECK(cudaMalloc(&send_buf, HOST_TOTAL_DATA_TO_CLIENT_SIZE));
	mr_send = ibv_reg_mr(pd, send_buf, HOST_TOTAL_DATA_TO_CLIENT_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
	if (!mr_send) {
		std::cerr << "ibv_reg_mr() failed for data_from_host" << std::endl;
		exit(1);
	}

	struct ibv_cq *recv_cq = ibv_create_cq(context, HOST_RECV_CQ_SIZE, NULL, NULL, 0);
    if (!recv_cq) {
    	printf("ERROR: ibv_create_cq() failed\n");
		exit(1);
    }

	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.send_cq = recv_cq;
	qp_init_attr.recv_cq = recv_cq;
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.cap.max_send_wr = 0;
	qp_init_attr.cap.max_recv_wr = HOST_MAX_RECV_WQES;
	qp_init_attr.cap.max_send_sge = 0;
	qp_init_attr.cap.max_recv_sge = 1;
//    qp_init_attr.cap.max_inline_data = 0;
	struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
	if (!qp) {
        std::cerr << "ibv_create_qp() failed errno= " << errno << std::endl;
		exit(1);
	}
    
	int ret;
	struct ibv_port_attr port_attr;
	ret = ibv_query_port(context, PORT_NUM, &port_attr);
	if (ret) {
		std::cerr << "ibv_query_port() failed ret=" << ret << std::endl;
		exit(1);
	}

	struct ib_info_t my_info;
	my_info.lid = port_attr.lid;
	my_info.qpn = qp->qp_num;
	my_info.mkey_response_buffer = mr_send->rkey;
	my_info.addr_response_buffer = (uintptr_t)mr_send->addr;
	int gid_index = get_gid_index(context);
	if (ibv_query_gid(context, 1, gid_index, &(my_info.gid) )) {
		std::cerr << "ibv_query_gid failed for gid " << gid_index << std::endl;
		exit(1);
	}
	
	ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "send" << std::endl;
		exit(1);
	}

	struct ib_info_t client_info;
	recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
	if (ret < 0) {
		std::cerr << "recv" << std::endl;
		exit(1);
	}

	struct ibv_qp_attr qp_attr;
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_INIT;
	qp_attr.pkey_index = 0;
	qp_attr.port_num = PORT_NUM;
	qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
	ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		std::cerr << "ibv_modify_qp() to INIT failed" << std::endl;
		exit(1);
	}
	
	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
	qp_attr.qp_state = IBV_QPS_RTR;
	qp_attr.path_mtu = IBV_MTU_4096;
	qp_attr.dest_qp_num = client_info.qpn;
	qp_attr.rq_psn      = 0 ;
	qp_attr.max_dest_rd_atomic = 1;
	qp_attr.min_rnr_timer = 12;
	qp_attr.ah_attr.is_global = 1;
	qp_attr.ah_attr.grh.dgid = client_info.gid;
	qp_attr.ah_attr.grh.sgid_index = get_gid_index(context);
	qp_attr.ah_attr.grh.flow_label = 0;
	qp_attr.ah_attr.grh.hop_limit = 1;
	qp_attr.ah_attr.grh.traffic_class = 0;
	qp_attr.ah_attr.dlid = client_info.lid;
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
		std::cerr << "ibv_modify_qp() to RTS failed" << std::endl;
		exit(1);
    }

	ib_resources->context = context;
	ib_resources->pd = pd;
	ib_resources->qp = qp;
	ib_resources->recv_cq = recv_cq;
	ib_resources->send_cq = recv_cq;
	ib_resources->lsend_buf = send_buf;
	ib_resources->lmr_send = mr_send;

	return ib_resources;
}


hostContext::hostContext(const string& interface, unsigned int workers_num, unsigned int tcp_port) : _workers_num(workers_num) {

	int lfd, sfd;
	int server_tcp_port = tcp_port;
	lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
    	std::cerr << "socket" << std::endl;
        exit(1);
    }
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(server_tcp_port);
	
	if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
    	std::cerr << "bind lfd" << std::endl;
        exit(1);
    }
    listen(lfd, 1);

	std::cout << "Host is waiting on port " << server_tcp_port << " to establish RX Queue. BlueField can connect." << std::endl;
	sfd = accept(lfd, NULL, NULL);
    if (sfd < 0) {
      	std::cerr << "accept sfd1" << std::endl;
       	exit(1);
   	}
	std::cout << "BlueField is connected" << std::endl;
	std::cout << "create RX Queue " << std::endl;
    recv_data_ib_resources = setup_recv_data_connection(interface,sfd);
	close(sfd);
	
	std::cout << "Host is waiting on port " << server_tcp_port << " to establish TX Queue. BlueField can connect." << std::endl;
    sfd = accept(lfd, NULL, NULL);
    if(sfd < 0) {
		std::cerr << "accept sfd" << std::endl;
        exit(1);
    }
	std::cout << "create TX Queue " << std::endl;
    send_data_ib_resources = setup_send_data_connection(interface,sfd);

	std::cout << "create Side Channel Notification " << std::endl;
	notify_ib_resources = setup_notify_connection(interface,sfd);
	close(sfd);

	close(lfd);

	_d_req_base_addresses = NULL;
	_d_resp_base_addresses = NULL;

}


hostContext::~hostContext() {
	std::cout << "kill hostcontext" << std::endl;
	teardown_connection(notify_ib_resources);
	teardown_connection(recv_data_ib_resources);
    teardown_connection(send_data_ib_resources);
	free(recv_data_ib_resources);
    free(send_data_ib_resources);
	if(_d_req_base_addresses != NULL){
		CUDA_CHECK(cudaFree(_d_req_base_addresses));
	}
	if(_d_resp_base_addresses != NULL){
        CUDA_CHECK(cudaFree(_d_resp_base_addresses));
    }
}


void* hostContext::getRequestBaseAddress() {
	return recv_data_ib_resources->lrecv_buf;
}

void* hostContext::getResponseBaseAddress() {
    return send_data_ib_resources->lsend_buf;
}


unsigned int* hostContext::getRequestCIBaseAddress() {
	return (unsigned int*) (notify_ib_resources->lsend_buf) + _workers_num;
}


unsigned int* hostContext::getResponsePIBaseAddress() {
    return (unsigned int*) notify_ib_resources->lsend_buf;
}


unsigned int* hostContext::getResponseCIBaseAddress() {
    return (unsigned int*) (notify_ib_resources->lrecv_buf);
}


void** hostContext::getDeviceReqBaseAddresses() {
    void *req_base_addresses = getRequestBaseAddress();
    CUDA_CHECK(cudaMalloc(&_d_req_base_addresses, sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(_d_req_base_addresses,&req_base_addresses, sizeof(void*), cudaMemcpyHostToDevice));
	return _d_req_base_addresses;
}


void** hostContext::getDeviceRespBaseAddresses() {
    void *resp_base_addresses = getResponseBaseAddress();
    CUDA_CHECK(cudaMalloc(&_d_resp_base_addresses, sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(_d_resp_base_addresses, &resp_base_addresses, sizeof(void*), cudaMemcpyHostToDevice));
	return _d_resp_base_addresses;
}

void hostContext::waitDevice(){
	CUDA_CHECK(cudaDeviceSynchronize());
}
