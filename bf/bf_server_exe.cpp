#include "bf_server.hpp"

CONNECTION_TYPE parse_connection_type(char* str) {
	if(!strcmp(str,"IB_CONNECTION")) {
		return IB_CONNECTION;
	}
	if(!strcmp(str,"UDP_CONNECTION")) {
        return UDP_CONNECTION;
    }
	if(!strcmp(str,"TCP_CONNECTION")) {
        return TCP_CONNECTION;
    }

	std::cerr << "unknown CONNECTION TYPE: " << str << std::endl;
	exit(1);
}


int main(int argc, char *argv[]) {
    unsigned int client_port = 5000;
	unsigned int host_port = 5000;
	CONNECTION_TYPE connection_type = IB_CONNECTION;
	unsigned int workers_num = 1;
	switch(argc) {
		case 1:
			std::cout << "No arguments were passed, use default values: " << std::endl;
			break;
		case 2:
			connection_type = parse_connection_type(argv[1]);
			break;
		case 3:
			connection_type = parse_connection_type(argv[1]);
			host_port = atoi(argv[2]);
			break;
		case 4:
			connection_type = parse_connection_type(argv[1]);
            host_port = atoi(argv[2]);
			client_port = atoi(argv[3]);
			break;
		case 5:
			connection_type = parse_connection_type(argv[1]);
            host_port = atoi(argv[2]);
            client_port = atoi(argv[3]);
			workers_num = atoi(argv[4]);
            break; 
		default:
			std::cerr << "Too many arguments: (" << argc - 1 << ") while expecting (max 5)." << std::endl;
    	    exit(1);
	}
	
	std::cout << "connection type:" << connection_type << "  host port: " << host_port << "  client port: " << client_port << "workers number: " << workers_num << std::endl;

    BFContext bf_ctx(connection_type, host_port, client_port, workers_num);
	bf_ctx.run_all();
    
	printf("Done\n");

    return 0;
}

