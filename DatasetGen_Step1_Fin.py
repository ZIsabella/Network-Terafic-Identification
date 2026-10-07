# Import required libraries:
#==================================================
#-------------------------
import os
import csv
#-------------------------

# [dpkt]:
#-------------------------
import dpkt
#-------------------------

# [numpy]:
#------------------------------
import numpy as np
#------------------------------

# [keras]:
#------------------------------
import keras
from   keras.utils.np_utils    import to_categorical
from   sklearn.model_selection import train_test_split
#------------------------------
#==================================================


# Parameters:
#==================================================
srcPath   = '../_pcap/X/'
dstPath   = './Dataset_out/'

X_Dim     = 1500
Y_Class   = 70

X_Data    = []
Y_Data    = []

LABEL_NUM = [0]*Y_Class

LABEL2DIG = {
    'WebBrowse_Chrome_None_IOS_FibreTCI'                   :0 ,
    'WebBrowse_Chrome_None_Android_FibreTCI'               :1 ,
    'WebBrowse_Chrome_None_Win7_FibreTCI'                  :2 ,
    'WebBrowse_Chrome_None_Win7_ITRC'                      :3 ,
    'WebBrowse_Chrome_UltraSurf_Win7_ITRC'                 :4 ,
    'WebBrowse_Firefox_None_Fedroa_FibreTCI'               :5 ,
    'WebBrowse_Firefox_None_Win7_MCI'                      :6 ,
    'WebBrowse_Firefox_None_Win10_MCI'                     :7 ,
    'WebBrowse_Firefox_None_Win7_ITRC'                     :8 ,
    'WebBrowse_Firefox_UltraSurf_Win7_ITRC'                :9 ,
    'WatchVideo_Instagram_None_IOS_FibreTCI'               :10,
    'WatchVideo_Instagram_None_Android_FibreTCI'           :11,
    'WatchVideo_Instagram_None_Fedroa_FibreTCI'            :12,
    'WatchVideo_Instagram_None_IOS_ITRC'                   :13,
    'WatchVideo_Firefox_None_Win7_ITRC'                    :14,
    'WatchVideo_Youtube_UltraSurf_IOS_FibreTCI'            :15,
    'WatchVideo_Youtube_UltraSurf_Android_FibreTCI'        :16,
    'WatchVideo_Youtube_UltraSurf_Win7_MCI'                :17,
    'WatchVideo_Youtube_UltraSurf_Win7_ITRC'               :18,
    'WatchVideo_Youtube_None_Android_FibreTCI'             :19,
    'WatchVideo_Youtube_None_Fedroa_FibreTCI'              :20,
    'WatchVideo_Chrome_None_Win7_ITRC'                     :21,
    'VoiceCall_Firefox_None_Fedroa_FibreTCI'               :22,
    'VideoCall_Skype_None_IOS_ITRC'                        :23,
    'VideoCall_Skype_None_Fedroa_FibreTCI'                 :24,
    'VideoCall_Whatsapp_None_IOS_FibreTCI'                 :25,
    'VideoCall_Instagram_None_IOS_ITRC'                    :26,
    'VideoCall_Instagram_UltraSurf_Android_FibreTCI'       :27,
    'VoiceCall_Skype_None_Fedroa_FibreTCI'                 :28,
    'VoiceCall_Whatsapp_None_IOS_FibreTCI'                 :29,
    'VoiceCall_Instagram_None_IOS_ITRC'                    :30,
    'VoiceCall_Instagram_UltraSurf_Android_FibreTCI'       :31,
    'SendEmail_Chrome_None_Fedroa_FibreTCI'                :32,
    'SendEmail_Chrome_None_Win7_ITRC'                      :33,
    'SendEmail_Firefox_None_Win7_ITRC'                     :34,
    'TextMessage_Skype_None_Fedroa_FibreTCI'               :35,
    'TextMessage_Skype_None_Win7_FibreTCI'                 :36,
    'TextMessage_Skype_UltraSurf_IOS_FibreTCI'             :37,
    'TextMessage_Skype_UltraSurf_Win7_FibreTCI'            :38,
    'TextMessage_Skype_HotspotShield_Win7_FibreTCI'        :39,
    'TextMessage_Whatsapp_None_Fedroa_FibreTCI'            :40,
    'FileDL_Chrome_None_Win7_ITRC'                         :41,
    'FileDL_Firefox_None_Win7_ITRC'                        :42,

    'WebBrowse_Chrome_Lentern_Fedroa_ITRC'                 :43,
    'WebBrowse_Firefox_Lentern_Fedroa_ITRC'                :44,
    'WebBrowse_Instagram_Lentern_Android_FibreTCI'         :45,
    'WatchVideo_Firefox_Lentern_Fedroa_ITRC'               :46,
    'WatchVideo_Instagram_Lentern_IOS_FibreTCI'            :47,
    'WatchVideo_Instagram_Lentern_Fedroa_FibreTCI'         :48,
    'WatchVideo_Youtube_Lentern_Android_FibreTCI'          :49,
    'WatchVideo_Youtube_Lentern_Fedroa_FibreTCI'           :50,
    'WatchVideo_Youtube_Lentern_Fedroa_ITRC'               :51,
    'VideoCall_Skype_Lentern_IOS_FibreTCI'                 :52,
    'VideoCall_Skype_Lentern_Fedroa_FibreTCI'              :53,
    'VideoCall_Skype_Lentern_IOS_ITRC'                     :54,
    'VideoCall_Skype_Lentern_Fedroa_ITRC'                  :55,
    'VideoCall_Instagram_Lentern_Android_FibreTCI'         :56,
    'VideoCall_Whatsapp_Lentern_IOS_FibreTCI'              :57,
    'VoiceCall_Skype_Lentern_Fedroa_FibreTCI'              :58,
    'VoiceCall_Skype_Lentern_IOS_ITRC'                     :59,
    'VoiceCall_Skype_Lentern_Fedroa_ITRC'                  :60,
    'VoiceCall_Whatsapp_Lentern_IOS_FibreTCI'              :61,
    'VoiceCall_Instagram_Lentern_Android_FibreTCI'         :62,
    'SendEmail_Chrome_Lentern_Fedroa_FibreTCI'             :63,
    'SendEmail_Firefox_Lentern_Fedroa_ITRC'                :64,
    'TextMessage_Skype_Lentern_Fedroa_FibreTCI'            :65,
    'TextMessage_Whatsapp_Lentern_Fedroa_FibreTCI'         :66,
    'FileDL_Chrome_Lentern_Fedroa_ITRC'                    :67,
    'FileDL_Firefox_Lentern_Fedroa_FibreTCI'               :68,
    'FileDL_Firefox_Lentern_Fedroa_ITRC'                   :69,
}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
#==================================================


# Data Information:
#==================================================
fileLabelDict   = {}
fileLabelDict_D = []

#------------------------------
fID       = open('../_pcap/DataInfo/discard_Fin.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    fileLabelDict_D.append(row[0])

fID.close()
#------------------------------

#------------------------------
fID       = open('../_pcap/DataInfo/discard_0.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    fileLabelDict_D.append(row[0])

fID.close()
#------------------------------

#------------------------------
fID       = open('../_pcap/DataInfo/discard_1.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    fileLabelDict_D.append(row[0])

fID.close()
#------------------------------

#------------------------------
fID       = open('../_pcap/DataInfo/datalog_Fin.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    if row[0] not in fileLabelDict_D:
        fileLabelDict[row[0]] = row[1]+'_'+row[2]+'_'+row[3]+'_'+row[4]+'_'+row[5]

fID.close()
#------------------------------

#------------------------------
fID       = open('../_pcap/DataInfo/datalog_0.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    if row[0] not in fileLabelDict_D:
        fileLabelDict[row[0]] = row[1]+'_'+row[2]+'_'+row[3]+'_'+row[4]+'_'+row[5]

fID.close()
#------------------------------

#------------------------------
fID       = open('../_pcap/DataInfo/datalog_1.csv')
csvreader = csv.reader(fID)

for row in csvreader:
    if row[0] not in fileLabelDict_D:
        fileLabelDict[row[0]] = row[1]+'_'+row[2]+'_'+row[3]+'_'+row[4]+'_'+row[5]

fID.close()
#------------------------------
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
for file in os.listdir(srcPath):
    if file.endswith('.pcap'):
        Path.append([os.path.join(srcPath, file), 0])
    if file.endswith('.pcapng'):
        Path.append([os.path.join(srcPath, file), 1])

Func = [dpkt.pcap.Reader, dpkt.pcapng.Reader]
#------------------------------

#------------------------------
for path in Path:
    #---------------
    fileName = path[0].split('/')[-1]
    if fileName not in fileLabelDict.keys():
        print("Discarded File: {}".format(fileName))
        continue
    #---------------

    #---------------
    fid         = [open(path[0],'rb'), path[0], fileLabelDict[fileName], path[1]]
    pcap_reader = Func[fid[3]](fid[0])
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
                    tp_hex           = Source_port + Destination_port + Window_size + Checksum + tp_hex_payload

            if (type(ip.data) == dpkt.udp.UDP):
                Source_port      = tp_hex_header[0  : 4 ]
                Destination_port = tp_hex_header[4  : 8 ]
                Length           = tp_hex_header[8  : 12]
                Checksum         = tp_hex_header[12 : 16]        
                tp_hex           = Source_port + Destination_port + Length + Checksum + tp_hex_payload

            data_hex = ip_hex_header + tp_hex
            #---------------
        
            #---------------
            data_byte = [int(data_hex[i:i+2],16) for i in range(0, min(len(data_hex),2*X_Dim), 2)]
            data_byte = data_byte + ([0]*(X_Dim-len(data_byte)))
            data_byte = np.array(data_byte).astype('uint8')

            if (fid[2] == 'VideoCall_Whatsapp_None_IOS_ITRC'):
                fid[2] =  'VideoCall_Whatsapp_None_IOS_FibreTCI'

            if ((fid[2] == 'TextMessage_Whatsapp_None_IOS_ITRC') or (fid[2] == 'TextMessage_Whatsapp_None_IOS_FibreTCI')):
                fid[2] =  'TextMessage_Whatsapp_None_Fedroa_FibreTCI'
            
            if (fid[2] == 'VideoCall_Skype_None_IOS_FibreTCI'):
                fid[2] =  'VideoCall_Skype_None_IOS_ITRC'

            if (fid[2] == 'VoiceCall_Skype_None_Win7_FibreTCI'):
                fid[2] =  'VoiceCall_Skype_None_Fedroa_FibreTCI'

            if (fid[2] == 'VoiceCall_Skype_None_Android_FibreTCI'):
                fid[2] =  'VoiceCall_Skype_None_Fedroa_FibreTCI'

            if (fid[2] == 'VoiceCall_Skype_None_IOS_FibreTCI'):
                fid[2] =  'VoiceCall_Skype_None_Fedroa_FibreTCI'

            if (fid[2] == 'TextMessage_Whatsapp_Lentern_IOS_FibreTCI'):
                fid[2] =  'TextMessage_Whatsapp_Lentern_Fedroa_FibreTCI'

            if (fid[2] == 'FileDL_Chrome_Lentern_Fedroa_FibreTCI'):
                fid[2] =  'FileDL_Chrome_Lentern_Fedroa_ITRC'

            if (fid[2] == 'FileDL_Chrome_None_Fedroa_FibreTCI'):
                fid[2] =  'FileDL_Chrome_None_Win7_ITRC'

            if (fid[2] == 'WebBrowse_Chrome_Lentern_Fedroa_FibreTCI'):
                fid[2] =  'WebBrowse_Chrome_Lentern_Fedroa_ITRC'

            if (fid[2] == 'WebBrowse_Chrome_None_Fedroa_FibreTCI'):
                fid[2] =  'WebBrowse_Chrome_None_Win7_FibreTCI'

            if (fid[2] == 'VideoCall_Skype_None_Android_FibreTCI'):
                fid[2] =  'VideoCall_Skype_None_Fedroa_FibreTCI'

            X_Data.append(data_byte.copy())
            Y_Data.append(LABEL2DIG[fid[2]])

            buffCnt += 1
            LABEL_NUM[LABEL2DIG[fid[2]]] += 1
            #---------------

    print('[{:3d}]-> Path: {:35s} | Category: {}, Num: {}, ARP_REMOVE: {}, DNS_REMOVE: {}, TCP_REMOVE: {}'.format(PathCnt, fid[1], fid[2], buffCnt, ARP_cnt, DNS_cnt, TCPHand_cnt))

    #---------------
    fid[0].close()
    fid         = []
    pcap_reader = []
    #---------------
#------------------------------

#------------------------------
Path = []
[print('Category: {:15s} | Num: {}'.format(DIG2LABEL[cIdx], LABEL_NUM[cIdx])) for cIdx in range(Y_Class)]
#------------------------------
#==================================================


#==================================================
if not os.path.exists(dstPath):
    os.mkdir(dstPath)

print(np.shape(X_Data))
np.save('./'+dstPath+'X_Data', np.asarray(X_Data))

print(np.shape(Y_Data))
np.save('./'+dstPath+'Y_Data', np.asarray(Y_Data))
#==================================================