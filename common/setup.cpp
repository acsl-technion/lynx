#include "setup.hpp"

double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}


struct ibv_context *ibv_open_device_by_name(const std::string& device_name)
{
    int num_devices = 0;
    struct ibv_device **devices_list = ibv_get_device_list(&num_devices);
    if(!devices_list){
            printf("ERROR: ibv_get_device_list() failed\n");
            exit(1);
    }
        std::cout << "ibv_open_device_by_name: " << num_devices << " devices were found." << std::endl;
        std::cout << "device_name is " << device_name << std::endl;
    for (int i = 0; i < num_devices; ++i) {
        string cur_name = ibv_get_device_name(devices_list[i]);
        std::cout << "device[" << i << "] name: " << cur_name << std::endl;
            if (device_name == cur_name){
                std::cout << device_name << " found." << std::endl;
                return ibv_open_device(devices_list[i]);
	    }
    }

    printf("ERROR: device named '%s' not found\n", device_name.c_str());
    return NULL;
}

string ib_device_from_netdev(const string& netdev)
{
    fs::path dir = "/sys/class/net/" + netdev + "/device/infiniband";
    fs::directory_iterator end;
    for (fs::directory_iterator dir_itr(dir); dir_itr != end; ++dir_itr) {
        return dir_itr->path().filename().string();
    }

    printf("Could not find IB device of netdev '%s'\n", netdev.c_str());
    abort();
}

int get_gid_index(ibv_context* dev)
{
        for (int i = 0; i < 0xffff; ++i) {
                ibv_gid gid;

                if (ibv_query_gid(dev, 1, i, &gid)) {
                        printf("ibv_query_gid failed for gid %d", i);
                        exit(1);
                }

                /* Check for IPv4 */
                if (gid.global.subnet_prefix != 0ull ||
                    (gid.global.interface_id & 0xffffffff) != 0xffff0000ull)
                        continue;

                char gid_type_str[7];
                int len = ibv_read_sysfs_file("/sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types",
                        boost::lexical_cast<string>(i).c_str(), gid_type_str, sizeof(gid_type_str));
                if (len < 0) {
                        printf("cannot read gid type for gid %d", i);
                        return -1;
                }

                if (strncasecmp(gid_type_str, "RoCE v2", len) != 0)
                        continue;

                /* TODO check also the netdev matches */
                return i;
        }
        return -1;
}

