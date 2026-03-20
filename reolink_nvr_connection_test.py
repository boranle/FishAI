from reolink_aio.api import Host
import asyncio

# Port numbers and their connections
# 9000 = basic network services (tested)
# 554 = RTSP (tested)
# 8000 = ONVIF (tested)

#TODO: TEST WITH MULTIPLE DEVICES ATTACHED TO NVR
print("Starting connection test")
async def print_mac_address():
    # initialize the host
    host = Host('192.168.1.100','admin', 'Fishfish', port=9000)
    print("Connection made")
    # connect and obtain/cache device settings and capabilities
    await host.get_host_data()
    # check if it is a camera or an NVR
    print(dir(host))
    #print(f"It is an NVR: {host.is_nvr}, number of channels: {host.num_channel}")
    # print mac address
    print(f"MAC: {host.mac_address}")
    # close the device connection
    await host.logout()

if __name__ == "__main__":
    asyncio.run(print_mac_address())