# Import required libraries:
#==================================================
#-------------------------
import os
#-------------------------

# [dpkt]:
#-------------------------
import dpkt
#-------------------------

# [numpy]:
#------------------------------
import numpy as np
#------------------------------
#==================================================


#------------------------------
root_path = './_pcap/test01/'
#------------------------------


# Parameters:
#==================================================
X_Dim     = 1500
X_Data    = []

tmp       = root_path.split('/')
tmp[1]    = '_preproc'
dest_path = os.path.join(*tmp)
#==================================================


# Data Extraction:
#==================================================
#------------------------------
Path    = []
PathCnt = 0

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW      = 101
#------------------------------

#------------------------------
for file in os.listdir(root_path):
    if file.endswith('.pcap'):
        Path.append([os.path.join(root_path, file), 0])
    if file.endswith('.pcapng'):
        Path.append([os.path.join(root_path, file), 1])

Func = [dpkt.pcap.Reader, dpkt.pcapng.Reader]
#------------------------------

#------------------------------
for path in Path:
    #---------------
    fid         = [open(path[0],'rb'), path[0], path[1]]
    pcap_reader = Func[fid[2]](fid[0])
    isRaw       = (pcap_reader.datalink() == LINKTYPE_RAW)
    isEthernet  = (pcap_reader.datalink() == LINKTYPE_ETHERNET)
    #---------------

    #---------------
    PathCnt    += 1
    buffCnt     = 0

    ARP_cnt     = 0
    DNS_cnt     = 0
    TCPHand_cnt = 0
    #---------------

    for ts, buf in pcap_reader:
        #---------------
        if isEthernet:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                ARP_cnt += 1
                continue
            ip = eth.data
        elif isRaw:
            ip = dpkt.ip.IP(buf)
        else:
            print("Unknown type!")
            continue
        #---------------

        try:
            #---------------
            dns = dpkt.dns.DNS(ip.data.data)
            DNS_cnt += 1
            continue
            #---------------
        except:
            #---------------
            ip_hex_header = bytes(ip).hex()[0:24]
            tp_hex        = bytes(ip.data).hex()

            tp_hex_header_len = len(bytes(ip.data).hex()) - len(bytes(ip.data.data).hex())
            tp_hex_header     = tp_hex[0:tp_hex_header_len]
            tp_hex_payload    = tp_hex[tp_hex_header_len:]
                        
            if (type(ip.data) == dpkt.tcp.TCP):
                if (len(bytes(ip.data.data).hex()) == 0):
                    TCPHand_cnt += 1
                    continue
                else:
                    Source_port	     = tp_hex_header[0  : 4 ]
                    Destination_port = tp_hex_header[4  : 8 ]
                    Sequence_num     = tp_hex_header[8  : 16]
                    Ack_num          = tp_hex_header[16 : 24]
                    Data_offset      = tp_hex_header[24 : 25]
                    Reserved_NS      = tp_hex_header[25 : 26]
                    Flags            = tp_hex_header[26 : 28]
                    Window_size      = tp_hex_header[28 : 32]
                    Checksum         = tp_hex_header[32 : 36]
                    Urgent_pointer   = tp_hex_header[36 : 40]
                    # tp_hex           = tp_hex_payload
                    tp_hex           = Window_size + Checksum + tp_hex_payload

            if (type(ip.data) == dpkt.udp.UDP):
                Source_port      = tp_hex_header[0  : 4 ]
                Destination_port = tp_hex_header[4  : 8 ]
                Length           = tp_hex_header[8  : 12]
                Checksum         = tp_hex_header[12 : 16]        
                # tp_hex           = tp_hex_payload    
                tp_hex           = Length + Checksum + tp_hex_payload    

            data_hex = ip_hex_header + tp_hex
            #---------------
        
            #---------------
            data_byte = [int(data_hex[i:i+2],16) for i in range(0, min(len(data_hex),2*X_Dim), 2)]
            data_byte = data_byte + ([0]*(X_Dim-len(data_byte)))
            data_byte = np.array(data_byte).astype('uint8')

            X_Data.append(data_byte.copy())

            buffCnt += 1
            #---------------

    print('[{:3d}]-> Path: {:32s} |  NUM_PACKET: {:6d}'.format(PathCnt, fid[1], buffCnt))
    # print('[{:3d}]-> Path: {:32s} |  NUM_PACKET: {:6d}  |  ARP_REMOVE: {:6d}  |  DNS_REMOVE: {:6d}  |  TCP_REMOVE: {:6d}'.format(PathCnt, fid[1], buffCnt, ARP_cnt, DNS_cnt, TCPHand_cnt))

    #---------------
    fid[0].close()
    fid         = []
    pcap_reader = []
    #---------------
#------------------------------

#------------------------------
Path = []
#------------------------------
#==================================================


#==================================================
if not os.path.exists(dest_path):
    os.mkdir(dest_path)

np.save('./'+dest_path+'X_Data', np.asarray(X_Data))
print('\n==> Total number of packets:', np.shape(X_Data)[0])
#==================================================