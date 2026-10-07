#-------------------------
import os
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

#------------------------------
Y_Class = 70
dstPath = './Lantern_Dataset/'

X_Data = np.load('./X_Data.npy')
Y_Data = np.load('./Y_Data.npy')

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
#------------------------------

# Data Storage:
#==================================================
#------------------------------
Y_Data = to_categorical(Y_Data, num_classes=Y_Class)
X_train, X_t,    y_train, y_t    = train_test_split(X_Data, Y_Data, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_t,    y_t,    test_size=0.50, random_state=42)

X_Data    = []
Y_Data    = []
X_t       = []
y_t       = []

# y_test = to_categorical(Y_Data, num_classes=Y_Class)
# X_test = X_Data
#------------------------------

#------------------------------
#-------------
Cat = [0]*Y_Class
Cat = np.array([Cat]*Y_Class)
Cat[range(Y_Class), range(Y_Class)] = 1
#-------------

#-------------
print('\nTraining:')
print('-'*20)

y_train_B = np.zeros((len(y_train), 2))

cnt = 0
for cat in Cat:
    idx  = np.where((y_train==cat).all(axis=1))
    size = np.shape(idx)[1]
    print("Category: {:55s} | Num: {}".format(DIG2LABEL[np.argmax(cat)], size))

    if (cnt < 43):
        replace = np.array([0.0, 1.0])
    else:
        replace = np.array([1.0, 0.0])

    y_train_B[idx] = replace
    cnt += 1

print("y_train 0: ", np.shape(np.where((y_train_B==np.array([1.0, 0.0])).all(axis=1)))[1])
print("y_train 1: ", np.shape(np.where((y_train_B==np.array([0.0, 1.0])).all(axis=1)))[1])
print('-'*20)
#-------------

#-------------
print('\nValidation:')
print('-'*20)

y_valid_B = np.zeros((len(y_valid), 2))

cnt = 0
for cat in Cat:
    idx  = np.where((y_valid==cat).all(axis=1))
    size = np.shape(idx)[1]
    print("Category: {:55s} | Num: {}".format(DIG2LABEL[np.argmax(cat)], size))

    if (cnt < 43):
        replace = np.array([0.0, 1.0])
    else:
        replace = np.array([1.0, 0.0])

    y_valid_B[idx] = replace
    cnt += 1

print("y_valid 0: ", np.shape(np.where((y_valid_B==np.array([1.0, 0.0])).all(axis=1)))[1])
print("y_valid 1: ", np.shape(np.where((y_valid_B==np.array([0.0, 1.0])).all(axis=1)))[1])
print('-'*20)
#-------------

#-------------
print('\nTesting:')
print('-'*20)

y_test_B = np.zeros((len(y_test), 2))

cnt = 0
for cat in Cat:
    idx  = np.where((y_test==cat).all(axis=1))
    size = np.shape(idx)[1]
    print("Category: {:55s} | Num: {}".format(DIG2LABEL[np.argmax(cat)], size))

    if (cnt < 43):
        replace = np.array([0.0, 1.0])
    else:
        replace = np.array([1.0, 0.0])

    y_test_B[idx] = replace
    cnt += 1

print("y_test 0: ", np.shape(np.where((y_test_B==np.array([1.0, 0.0])).all(axis=1)))[1])
print("y_test 1: ", np.shape(np.where((y_test_B==np.array([0.0, 1.0])).all(axis=1)))[1])
print('-'*20)
#-------------
#------------------------------

#------------------------------
if not os.path.exists(dstPath):
    os.mkdir(dstPath)

print(np.shape(X_train))
np.save(dstPath+'X_train', np.asarray(X_train))

print(np.shape(X_valid))
np.save(dstPath+'X_valid', np.asarray(X_valid))

print(np.shape(X_test))
np.save(dstPath+'X_test', np.asarray(X_test))

print(np.shape(y_train_B))
np.save(dstPath+'y_train', np.asarray(y_train_B))

print(np.shape(y_valid_B))
np.save(dstPath+'y_valid', np.asarray(y_valid_B))

print(np.shape(y_test_B))
np.save(dstPath+'y_test', np.asarray(y_test_B))
#------------------------------
#==================================================