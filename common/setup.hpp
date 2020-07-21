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

#ifndef __SETUP_H__
#define __SETUP_H__

#include <iostream>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <assert.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/asio.hpp>

extern "C" {
#include <infiniband/driver.h>
}

#define N 10

#define PORT_NUM 1
#define DATA_QP_SIZE (32*1024)
#define CONTROL_QP_SIZE 4

#define TCP_PORT_NUM 5000
#define UDP_PORT_NUM 5000


#define SOCKPERF 

#ifdef SOCKPERF
	#define SOCKPERF_HEADER 16
#else
	#define SOCKPERF_HEADER 0
#endif


using boost::asio::local::stream_protocol;
using std::string;
namespace fs = boost::filesystem;

#define ACTUAL_PAYLOAD (28*28)

#define RECV_WQES_NUM (8*1024)
#define NOTIFY_WQES_NUM (32)
#define OWNER_INT_SIZE (sizeof(unsigned int))

#define CLIENT_MAX_RECV_WQES RECV_WQES_NUM
#define CLIENT_MAX_SEND_WQES CLIENT_MAX_RECV_WQES
#define CLIENT_RECV_CQ_SIZE CLIENT_MAX_RECV_WQES
#define CLIENT_SEND_CQ_SIZE CLIENT_MAX_SEND_WQES
#define CLIENT_SEND_MSG_SIZE (ACTUAL_PAYLOAD + SOCKPERF_HEADER)
#define CLIENT_RECV_MSG_SIZE (ACTUAL_PAYLOAD + SOCKPERF_HEADER)
#define CLIENT_TOTAL_DATA_TO_SERVER_SIZE (CLIENT_MAX_SEND_WQES * CLIENT_SEND_MSG_SIZE)
#define CLIENT_TOTAL_DATA_FROM_SERVER_SIZE (CLIENT_MAX_RECV_WQES * CLIENT_RECV_MSG_SIZE)
#define CLIENT_NUM_OF_WQE_LISTS (2)

#define HOST_MAX_RECV_WQES RECV_WQES_NUM
#define HOST_MAX_SEND_WQES HOST_MAX_RECV_WQES
#define HOST_RECV_CQ_SIZE HOST_MAX_RECV_WQES
#define HOST_SEND_CQ_SIZE HOST_MAX_SEND_WQES
#define HOST_SEND_MSG_SIZE CLIENT_RECV_MSG_SIZE 
#define HOST_RECV_MSG_SIZE CLIENT_SEND_MSG_SIZE
#define HOST_TOTAL_DATA_TO_CLIENT_SIZE (HOST_MAX_SEND_WQES * HOST_SEND_MSG_SIZE)
#define HOST_TOTAL_DATA_FROM_CLIENT_SIZE (HOST_MAX_RECV_WQES * (HOST_RECV_MSG_SIZE + OWNER_INT_SIZE))

//H2C - Host to Client
#define BF_MAX_RECV_WQES RECV_WQES_NUM
#define BF_MAX_SEND_WQES HOST_MAX_RECV_WQES
#define BF_RECV_CQ_SIZE BF_MAX_RECV_WQES
#define BF_SEND_CQ_SIZE BF_MAX_SEND_WQES
#define BF_C2H_MSG_SIZE (HOST_RECV_MSG_SIZE + OWNER_INT_SIZE) // owner int - it should be owner bit (alignment in GPU issues)
#define BF_H2C_MSG_SIZE HOST_SEND_MSG_SIZE
#define BF_TOTAL_DATA_TO_HOST_SIZE (BF_MAX_SEND_WQES * BF_C2H_MSG_SIZE)
#define BF_TOTAL_DATA_FROM_HOST_SIZE (BF_MAX_RECV_WQES * BF_H2C_MSG_SIZE)
#define BF_NUM_OF_WQE_LISTS (2)
#define BF_MAX_POLL_CQES (1)

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)



#define MOD(X,Y) ((X) & ((Y) - 1)) // Y has to be power of 2
//#define HAS_REQUEST(pi_val, ci_val, stride) ( ((ci_val <= pi_val)&&(ci_val+stride <= pi_val)) || \
                                        ((ci_val > pi_val)&&((ci_val+stride < MAX_WQES_NUMBER_PER_QP)||(MOD(ci_val+stride,MAX_WQES_NUMBER_PER_QP)<=pi_val))) )
//#define CAN_PUSH(pi_val, ci_val) ( MOD(pi_val+1,MAX_WQES_NUMBER_PER_QP) != ci_val)

#define CAN_PUSH(pi, ci, N) (FREE_SLOTS(pi, ci, N) > 0)
//#define HAS_REQUEST(pi, ci, N) (OCCUPIED_SLOTS(pi, ci, N) > 0)
#define HAS_REQUEST(pi, ci, N) (OCCUPIED_SLOTS(pi, ci, N))

#define FREE_SLOTS(pi, ci, N) ( ((pi) >= (ci)) ? ( (N) - ( (pi) - (ci) + 1 ) ) : ( (ci) - (pi) - 1) )
#define OCCUPIED_SLOTS(pi, ci, N) ( (ci) > (pi) ? ( (N) + (pi) - (ci) ) : ( (pi) - (ci)) )

struct ib_resources_t {
	unsigned int posted_wqes;
	unsigned int* load_factor;
	unsigned int resp_sent;
	unsigned int client_fd;
	unsigned int recv_buf_offset;
	unsigned int wrap_around;	
	bool update_wrap_around;
	
    struct ibv_context *context;
    struct ibv_pd *pd;

    struct ibv_qp *qp;
    struct ibv_cq *recv_cq;
    struct ibv_cq *send_cq;

    char* lrecv_buf;
    struct ibv_mr* lmr_recv;
    char* lsend_buf;
    struct ibv_mr* lmr_send;

    int rmr_recv_key;
    long long rmr_recv_addr;
    int rmr_send_key;
    long long rmr_send_addr;
};

struct ib_info_t {
    int lid;
    int qpn;
    int mkey_data_buffer;
    long long addr_data_buffer;
    int mkey_response_buffer;
    long long addr_response_buffer;
    ibv_gid gid;
};


int get_gid_index(ibv_context* dev);
struct ibv_context *ibv_open_device_by_name(const std::string& device_name);
string ib_device_from_netdev(const string& netdev);
double static inline get_time_msec();
#endif
