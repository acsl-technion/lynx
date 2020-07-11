#ifndef __BF_SERVER_H__
#define __BF_SERVER_H__

#include "bf_context.hpp"

class BFServer {
	unsigned int _connections_num;
	CONNECTION_TYPE _connection_type;
	BFContext** _bf_ctxs;

public:

	BFServer(unsigned int connections_base_id = 0, unsigned int connections_num = 1, CONNECTION_TYPE = IB_CONNECTION);
	~BFServer();
	void doLoop();	

};

#endif
