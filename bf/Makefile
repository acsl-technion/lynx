CC := g++ -std=c++11 -O3
CFLAGS=-Wall -g
LIBS := -libverbs -lrt -lpthread -lboost_filesystem -lboost_system

########################################################################

all: bf_server_exe

# link
bf_server_exe: setup.o bf_context.o bf_server.o bf_server_exe.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

########################################################################

# compile
bf_server_exe.o: ../common/setup.hpp bf_context.hpp bf_context.cpp bf_server_exe.cpp
	$(CC) $(CFLAGS) -c bf_server_exe.cpp

bf_server.o: ../common/setup.hpp bf_context.hpp bf_server.cpp
	$(CC) $(CFLAGS) -c bf_server.cpp

bf_context.o: ../common/setup.hpp bf_context.hpp bf_context.cpp
	$(CC) $(CFLAGS) -c bf_context.cpp

setup.o: ../common/setup.hpp ../common/setup.cpp
	$(CC) $(CFLAGS) -c ../common/setup.cpp

########################################################################
clean:
	\rm -f *.o bf_server_exe

