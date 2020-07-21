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


#include "bf_server.hpp"

BFServer::BFServer(unsigned int base_connection_id, unsigned int connections_num, CONNECTION_TYPE connection_type) : _connections_num(connections_num) , _connection_type(connection_type){
/*	_bf_ctxs = (BFContext**) malloc(_connections_num * sizeof(BFContext*));
	int sfd = -1;
	for(int i = 0 ; i < _connections_num ; i++) {
		bool first_connection = i == 0;
		bool last_connection = i == (_connections_num - 1);
		_bf_ctxs[i] = new BFContext(first_connection, last_connection, base_connection_id, i, _connection_type, _connections_num, sfd);
		sfd = _bf_ctxs[i]->get_host_sfd();
		if(i > 0) {
			_bf_ctxs[i]->set_nQP_ib_resources(_bf_ctxs[0]->get_nQP_ib_resources());
		}
	}*/
}


BFServer::~BFServer() {
	for(int i = 0 ; i < _connections_num ; i++) {
		delete(_bf_ctxs[i]);		
	}
	free(_bf_ctxs);
}
	
void BFServer::doLoop() {
	for(int i = 0 ; i < _connections_num ; i++) {
		//_bf_ctxs[i]->doLoop();		
	}
}
