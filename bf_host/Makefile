CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -Xptxas="-v" #-lineinfo
CC := g++ 

CFLAGS=-Wall -g 
NVCCFLAGS := -arch=sm_35 --device-c -g

NVCCLFLAGS := -arch=sm_35 

LDFLAGS := -I$(CUDA_PATH)/include -L$(CUDAPATH)/lib64 #-I~/gdrcopy

LIBPATH := $(CUDA_PATH)/lib64
SOLIBS := $(LIBPATH)/libcublas.so $(LIBPATH)/libcudart.so #~/gdrcopy/libgdrapi.so

LIBS := -libverbs -lrt -lpthread -lboost_filesystem -lboost_system

########################################################################

all: echo matrix_mul hello_kernel lenet busy_wait

busy_wait: setup.o bf_host.o busy_wait.o
	$(NVCC) $(NVCCLFLAGS) $(LDFLAGS) -o $@ $^ $(SOLIBS) $(LIBS)

echo: setup.o bf_host.o echo.o
	$(NVCC) $(NVCCLFLAGS) $(LDFLAGS) -o $@ $^ $(SOLIBS) $(LIBS)

matrix_mul: setup.o bf_host.o matrix_mul.o
	$(NVCC) $(NVCCLFLAGS) $(LDFLAGS) -o $@ $^ $(SOLIBS) $(LIBS)

hello_kernel: setup.o bf_host.o hello_kernel.o
	$(NVCC) $(NVCCLFLAGS) $(LDFLAGS) -o $@ $^ $(SOLIBS) $(LIBS)

lenet: setup.o bf_host.o lenet.o
	$(NVCC) $(NVCCLFLAGS) $(LDFLAGS) -o $@ $^ $(SOLIBS) $(LIBS)

########################################################################

# compile

busy_wait.o: gpu_define.cu.h ../common/setup.hpp busy_wait.cu
	$(NVCC) $(NVCCFLAGS) -c busy_wait.cu

echo.o: gpu_define.cu.h ../common/setup.hpp echo.cu
	$(NVCC) $(NVCCFLAGS) -c echo.cu

matrix_mul.o: gpu_define.cu.h ../common/setup.hpp matrix_mul.cu
	$(NVCC) $(NVCCFLAGS) -c matrix_mul.cu

hello_kernel.o: gpu_define.cu.h ../common/setup.hpp hello_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c hello_kernel.cu

bf_host.o: gpu_define.cu.h ../common/setup.hpp bf_host.cu.hpp bf_host.cu
	$(NVCC) $(NVCCFLAGS) -c bf_host.cu

setup.o: ../common/setup.hpp ../common/setup.cpp
	$(CC) $(CFLAGS) -c ../common/setup.cpp

lenet.o: gpu_define.cu.h ../common/setup.hpp lenet.cu
	$(NVCC) $(NVCCFLAGS) -c lenet.cu


########################################################################
clean:
	\rm -f *.o matrix_mul hello_kernel echo lenet
