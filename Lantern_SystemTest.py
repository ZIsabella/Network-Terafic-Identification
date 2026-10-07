# Import required libraries:
#==================================================
# [General]:
#-------------------------
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#-------------------------

# [numpy]:
#-------------------------
import numpy  as np
from   numpy.core.numeric import Infinity
#-------------------------

# [tensorflow]:
#-------------------------
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
#-------------------------

# [Version]:
#-------------------------
# print('- numpy version:      {}'.format(np.__version__))
# print('- tensorflow version: {}'.format(tf.__version__))
#-------------------------
#==================================================


#------------------------------
root_path = './_preproc/test01/'
#------------------------------


# Layer Wrappers:
#==================================================
# [Conv1d]:
#-------------------------
def conv1d(x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1,1,strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
#-------------------------

# [MaxPool2d]:
#-------------------------
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,1,k,1], strides=[1,1,k,1], padding='SAME')
#-------------------------

# [FullyConnected]:
#-------------------------
def fullyConnected(x, weight, biase, batch_normalization=False):
    x = tf.add(tf.matmul(x, weight), biase)
    if (batch_normalization==True):
        x = tf.layers.batch_normalization(x)
    return tf.nn.relu(x)
#-------------------------
#==================================================


# Models:
#==================================================
# [neural_net_p]:
#-------------------------
def neural_net_p(x, weights, biases_1, biases_2, dropout):
    layer     = fullyConnected(x, weights, biases_1, batch_normalization=False)
    layer     = tf.layers.dropout(layer, dropout)

    weights_t = tf.transpose(weights, name="w_t")
    out_layer = fullyConnected(layer, weights_t, biases_2)

    return layer, out_layer
#-------------------------

# [neural_net_stack]:
#-------------------------
def neural_net_stack(x, weights, biases, dropout):
    layer = [x]
    for eIdx in range(l_enc-1):
        layer.append(fullyConnected(layer[-1], weights['w'+str(eIdx+1)], biases['b'+str(eIdx+1)], batch_normalization=False))
        layer[-1] = tf.layers.dropout(layer[-1], dropout)

    for dIdx in range(l_enc):
        idx       = (l_enc-1) - dIdx 
        weights_t = tf.transpose(weights['w'+str(idx)], name='w_'+str(idx)+'t')
        layer.append(fullyConnected(layer[-1], weights_t, biases['b'+str(idx)+'_t']))
        if (dIdx != (l_enc-1)):
            layer[-1] = tf.layers.dropout(layer[-1], dropout)

    return layer[l_enc-1], layer[-1]
#-------------------------

# [neural_net_class]:
#-------------------------
def neural_net_class(x, weights, biases):
    x = tf.reshape(x, shape=[-1,1,64,1])

    conv1 = conv1d(x, weights['w_c_'+str(0)], biases['b_c_'+str(0)])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv1d(conv1, weights['w_c_'+str(1)], biases['b_c_'+str(1)])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['w_c_2'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w_c_2']), biases['b_c_2'])
    fc1 = tf.nn.relu(fc1)

    out_layer_class = tf.add(tf.matmul(fc1, weights['w'+str(l_enc)]), biases['b'+str(l_enc)])

    return out_layer_class
#-------------------------
#==================================================


# System Parameters:
#==================================================
num_input          = None
n_encLayer         = [416, 320, 256, 192, 128, 64]
convLayer          = [[1, 3, 1, 8], [1, 3, 8, 16]]
num_classes        = 2
init_learning_rate = 1e-3
l2_reg             = 0.0005
dropout_0          = 0.00

l_enc = len(n_encLayer)
n_hidden = n_encLayer.copy()
for hIdx in range(l_enc-1):
    n_hidden.append(n_encLayer[-(hIdx+2)])
#==================================================


# Prepare Dataset:
#==================================================
#-------------------------
X_Data = np.load(root_path + 'X_Data.npy').astype('float32')
X_Data = np.delete(X_Data,  np.s_[1480:], 1)
X_Data = X_Data/255
#-------------------------

#-------------------------
numOfPackets = np.shape(X_Data)[0]
inputDim     = np.shape(X_Data)[1]
n_hidden     = [inputDim] + n_hidden
#-------------------------
#==================================================


# Store Layers Weight & Bias:
#==================================================
initializer = tf.contrib.layers.variance_scaling_initializer()

# [Weights]:
#-------------------------
weights = dict()

for wIdx in range(l_enc):
    weights['w'+str(wIdx)] = tf.Variable(initializer([n_hidden[wIdx], n_hidden[wIdx+1]]), dtype=tf.float32, name='w_'+str(wIdx))

for wIdx in range(len(convLayer)):
    weights['w_c_'+str(wIdx)] = tf.Variable(initializer(convLayer[wIdx]), dtype=tf.float32, name='w_c_'+str(wIdx))

weights['w_c_2']        = tf.Variable(initializer([int(n_hidden[l_enc]/4)*16, n_hidden[l_enc]]), dtype=tf.float32, name='w_c_2')
weights['w'+str(l_enc)] = tf.Variable(initializer([n_hidden[l_enc], num_classes]), dtype=tf.float32, name='w_'+str(l_enc))
#-------------------------

# [Biases]:
#-------------------------
biases  = dict()

for bIdx in range(l_enc):
    biases['b'+str(bIdx)]      = tf.Variable(tf.zeros(n_hidden[bIdx+1]), name='b_'+str(bIdx))
    biases['b'+str(bIdx)+'_t'] = tf.Variable(tf.zeros(n_hidden[bIdx]),   name='b_'+str(bIdx)+'t')

for bIdx in range(len(convLayer)):
    biases['b_c_'+str(bIdx)]  = tf.Variable(tf.random_normal([convLayer[bIdx][3]]), name='b_c_'+str(bIdx))

biases['b_c_2']        = tf.Variable(tf.random_normal([n_hidden[l_enc]]), name='b_c_2')
biases['b'+str(l_enc)] = tf.Variable(tf.zeros(num_classes), name='b_'+str(l_enc))
#-------------------------
#==================================================


# Structure:
#==================================================
# [Placeholders]
#-------------------------
S = []
for sIdx in range(l_enc):
    S.append(tf.placeholder(tf.float32, shape=[None, n_hidden[sIdx]]))
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

drop_prob = tf.placeholder(tf.float32)
#-------------------------

# [Construct Model]
#-------------------------
hlayer      = [0]*l_enc
out_layer_p = [0]*l_enc

for hIdx in range(l_enc):
    hlayer[hIdx], out_layer_p[hIdx] = neural_net_p(S[hIdx], weights['w'+str(hIdx)], biases['b'+str(hIdx)], biases['b'+str(hIdx)+'_t'], drop_prob)
hlayer_stack, out_layer_stack = neural_net_stack(hlayer[0], weights, biases, drop_prob)
#-------------------------

# [Stacked AutoEncoder]
#-------------------------
regularizer           = tf.contrib.layers.l2_regularizer(l2_reg)
reconstruction_loss_p = []
reg_loss_p            = []
loss_op_p             = []
optimizer_p           = []
train_op_p            = []

for hIdx in range(l_enc):
    reconstruction_loss_p.append(tf.reduce_mean(tf.square(out_layer_p[hIdx] - S[hIdx])))
    reg_loss_p.append(regularizer(weights['w'+str(hIdx)]))
    loss_op_p.append(reconstruction_loss_p[hIdx] + reg_loss_p[hIdx])
    optimizer_p.append(tf.train.AdamOptimizer(learning_rate=init_learning_rate))
    train_op_p.append(optimizer_p[hIdx].minimize(loss_op_p[hIdx]))

reconstruction_loss_stack = tf.reduce_mean(tf.square(out_layer_stack - S[0]))
reg_loss_stack            = sum(regularizer(weights['w'+str(wIdx)]) for wIdx in range(l_enc))
loss_op_stack             = reconstruction_loss_stack + reg_loss_stack
optimizer_stack           = tf.train.AdamOptimizer(learning_rate=init_learning_rate)
train_op_stack            = optimizer_stack.minimize(loss_op_stack)
#-------------------------

# [Classifier]
#-------------------------
logits       = neural_net_class(hlayer_stack, weights, biases)
prediction   = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#-------------------------

#-------------------------
# Start training
saver = tf.train.Saver()
#-------------------------
#==================================================


# Results:
#==================================================
print("---[Packet Classifier System Started]---\n")
print('==> Total number of packets: {}'.format(numOfPackets))

sess = tf.Session()
with tf.Session() as sess:
    saver.restore(sess, './_model/v3.0/model.ckpt')
    Y_Pred = sess.run(prediction, feed_dict={S[0]: X_Data, drop_prob: dropout_0})

numOfLanternPackets      = sum(Y_Pred.argmax(axis = -1)==0)
numOfNonLanternPackets   = sum(Y_Pred.argmax(axis = -1)==1)

lanternPacketsPercent    = (numOfLanternPackets   /numOfPackets)*100
nonLanternPacketsPercent = (numOfNonLanternPackets/numOfPackets)*100

print("    - Number of \'Lantern\'     packets: {:7d} ({:5.3f}%)".format(numOfLanternPackets,    lanternPacketsPercent))
print("    - Number of \'Non-Lantern\' packets: {:7d} ({:5.3f}%)".format(numOfNonLanternPackets, nonLanternPacketsPercent))
#==================================================
