#!/bin/bash

LIBVMA=/root/libvma/src/vma/.libs/libvma.so

if [[ $1 == "-h" ]]; then
	echo "Usage: " $0 connection_type connections_num host_port client_port total_workers_num
	echo "       connection_type: UDP_CONNECTION [dfault = UDP_CONNECTION]" 
	echo "		 connections_num: [defualt = 1]"
	echo "       host port: [default = 5000]"
	echo "		 client port: [default = 5000]"
	echo "		 total workers number: [default = connections_num]"
	exit 1
fi

connection_type=UDP_CONNECTION
connections_num=1
host_port=5000
client_port=5000
total_workers_num=$connections_num

if [ $# -gt 0 ] ; then
	connection_type=$1; shift
fi
if [ $# -gt 0 ] ; then
    connections_num=$1; shift
fi
if [ $# -gt 0 ] ; then
    host_port=$1; shift
fi
if [ $# -gt 0 ] ; then
	client_port=$1; shift
fi
if [ $# -gt 0 ] ; then
	total_workers_num=$1; shift
	if [[ $total_workers_num -lt $connections_num ]]; then
		echo "number of workers is less than number of connections.."
		exit 1
	fi
fi


delta_workers=$[$total_workers_num/$connections_num]

env_vars=(\
    LD_PRELOAD=$LIBVMA \
	VMA_MTU=200 \
	VMA_RX_BUFS=500000 \
	VMA_RING_ALLOCATION_LOGIC_RX=20 \
	VMA_RX_POLL=1000 \
	VMA_RX_UDP_POLL_OS_RATIO=100 \
	VMA_SELECT_POLL=1000 \
	VMA_THREAD_MODE=1 \
	VMA_RX_POLL_YIELD=1 \
	VMA_RX_NO_CSUM=1\
	VMA_NICA_ACCESS_MODE=0 \
)

#sudo taskset -c $i env "${env_vars[@]}" ./bf_server_exe $connection_type $host_port $client_port
for i in `seq 0 $[$connections_num - 1]`; do
	workers_per_connection=$[$total_workers_num/$connections_num]
	temp=$[$total_workers_num % connections_num]
	if [[ $temp -gt $i ]]; then
		workers_per_connection=$[$workers_per_connection + 1]
	fi
	
	env "${env_vars[@]}" ./bf_server_exe $connection_type $host_port $client_port $workers_per_connection &
	#./bf_server_exe $connection_type $host_port $client_port $workers_per_connection &
	sleep 2s
	client_port=$[$client_port + 1]
	host_port=$[$host_port + 1]
done
