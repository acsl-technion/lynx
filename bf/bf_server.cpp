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
