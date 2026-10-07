# Import required libraries:
#==================================================
# [General]:
#-------------------------
import os
import logging
import itertools
#-------------------------

# [numpy]:
#-------------------------
import numpy  as np
from   numpy.core.numeric import Infinity
#-------------------------

# [matplotlib]:
#-------------------------
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#-------------------------

# [sklearn]:
#-------------------------
import sklearn
from   sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from   sklearn.metrics import recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
#-------------------------

# [tensorflow]:
#-------------------------
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
#-------------------------

# [Version]:
#-------------------------
print('- numpy version:      {}'.format(np.__version__))
print('- matplotlib version: {}'.format(matplotlib.__version__))
print('- sklearn version:    {}'.format(sklearn.__version__))
print('- tensorflow version: {}'.format(tf.__version__))
#-------------------------
#==================================================


# Functions:
#==================================================
# [nextBatch]:
# Return a total of maximum "num" random samples and labels.
# NOTE: The last batch will be of size len(data) % num
#-------------------------
def nextBatch(num, data, labels):
    num_el = data.shape[0]

    while True: # or whatever condition you may have
        idx = np.arange(0 , num_el)
        np.random.shuffle(idx)

        current_idx = 0
        while current_idx < num_el:
            batch_idx      = idx[current_idx:current_idx+num]
            current_idx   += num
            data_shuffle   = [data[ i,:] for i in batch_idx]
            labels_shuffle = [labels[ i] for i in batch_idx]
            yield np.asarray(data_shuffle), np.asarray(labels_shuffle)
#-------------------------

# [Plot Confusion Matrix]:
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting "normalize=True".
#-------------------------
def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize = (4,4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt    = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
#-------------------------
#==================================================


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
# Training parameters
#-------------------------
init_learning_rate = 1e-3
l2_reg             = 0.0005

n_epochs           = [20, 20, 20, 20, 20, 20, 1000]
n_patience         = [30, 30, 30, 30, 30, 30, 30]

batch_size         = 1024
batch_size_SAE     = 1024

dropout            = 0.20
dropout_0          = 0.00
#-------------------------

# Network Parameters
#-------------------------
num_input   = None
n_encLayer  = [416, 320, 256, 192, 128, 64]
convLayer   = [[1, 3, 1, 8], [1, 3, 8, 16]]
num_classes = 2

l_enc = len(n_encLayer)

n_hidden = n_encLayer.copy()
for hIdx in range(l_enc-1):
    n_hidden.append(n_encLayer[-(hIdx+2)])

LABEL2DIG = {'lantern':0, 'non-lantern':1}
DIG2LABEL = {v: k for k, v in LABEL2DIG.items()}
#-------------------------
#==================================================


# Setup the Parameters:
#==================================================
#-------------------------
np.random.seed(0)
tf.set_random_seed(0)
#-------------------------

#-------------------------
FOLDER = '/content/drive/MyDrive/OUTPUT_c'
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

MODEL_PATH   = FOLDER + '/model.ckpt'
MODEL_PATH_S = FOLDER + '/model_Stack.ckpt'
MODEL_PATH_A = FOLDER + '/model_Stack_All.ckpt'
FIG_PATH     = FOLDER + '/Confusion_Matrix.png'
FIG_PATH_N   = FOLDER + '/Confusion_Matrix_Norm.png'
#-------------------------
#==================================================


# Prepare Dataset:
#==================================================
#-------------------------
X_train = np.load('./X_train.npy')
X_train = np.delete(X_train, np.s_[1484:], 1)
X_train = np.delete(X_train, np.s_[12:12+4], 1)

X_valid = np.load('./X_valid.npy')
X_valid = np.delete(X_valid, np.s_[1484:], 1)
X_valid = np.delete(X_valid, np.s_[12:12+4], 1)

X_test  = np.load('./X_test.npy')
X_test  = np.delete(X_test,  np.s_[1484:], 1)
X_test  = np.delete(X_test,  np.s_[12:12+4], 1)

y_train = np.load('./y_train.npy').astype('float32')
y_valid = np.load('./y_valid.npy').astype('float32')
y_test  = np.load('./y_test.npy').astype('float32')
#-------------------------

#-------------------------
#-------------
Cat = [0]*num_classes
Cat = np.array([Cat]*num_classes)
Cat[range(num_classes), range(num_classes)] = 1

clw     = [0]*num_classes
maxsize = 0
#-------------

#-------------
print('-'*20)
for cat in Cat:
    size = np.shape(np.where((y_train==cat).all(axis=1)))[1]
    print(DIG2LABEL[np.argmax(cat)] + ": " + str(size))
    if (size!=0):
      clw[np.argmax(cat)] = 1/size
      if (size > maxsize):
          maxsize = size
print('-'*20)

clw = [i*maxsize for i in clw]
#-------------

#-------------
print('Training:')
print('-'*20)
for cat in Cat:
    size = np.shape(np.where((y_train==cat).all(axis=1)))[1]
    print(DIG2LABEL[np.argmax(cat)] + ": " + str(size))
print('-'*20)
#-------------

#-------------
print('Validation:')
print('-'*20)
for cat in Cat:
    size = np.shape(np.where((y_valid==cat).all(axis=1)))[1]
    print(DIG2LABEL[np.argmax(cat)] + ": " + str(size))
print('-'*20)
#-------------

#-------------
print('Testing:')
print('-'*20)
for cat in Cat:
    size = np.shape(np.where((y_test==cat).all(axis=1)))[1]
    print(DIG2LABEL[np.argmax(cat)] + ": " + str(size))
print('-'*20)
#-------------
#-------------------------

#-------------------------
size          = np.shape(X_train)[0]
dim           = np.shape(X_train)[1]
n_batches     = int(size/batch_size)
n_batches_SAE = int(size/batch_size_SAE)

num_input = dim
n_hidden  = [num_input] + n_hidden

print("num_input: ", num_input)
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
global_step = tf.Variable(0, trainable=False)

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
#-------------
logits       = neural_net_class(hlayer_stack, weights, biases)
prediction   = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#-------------
    
# Define loss:
#-------------
class_weights     = tf.constant([clw])
weights_x         = tf.reduce_sum(class_weights*Y, axis=1)
unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
weighted_losses   = unweighted_losses * weights_x
reg_loss_class    = sum(regularizer(value) for key,value in weights.items()) - regularizer(weights['w'+str(l_enc)])
loss_op_class     = tf.reduce_mean(weighted_losses)
#-------------

# Define optimizer:
#-------------
learning_rate   = tf.train.exponential_decay(init_learning_rate, global_step=global_step, decay_steps=60000, decay_rate=0.8, staircase=True)
add_global      = global_step.assign_add(1)
optimizer_class = tf.train.AdamOptimizer(learning_rate=learning_rate)

with tf.control_dependencies([add_global]):
    train_op_class = optimizer_class.minimize(loss_op_class+reg_loss_class)
#-------------
#-------------------------

#-------------------------
# Start training
saver = tf.train.Saver()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
#-------------------------
#==================================================


# Training:
#==================================================
with tf.Session() as sess:
    # [Run the initializer]
    #-------------------------
    sess.run(init)
    #-------------------------

    # [PART I]
    #-------------------------
    #---------------
    S_train = X_train.astype('float32')/255
    # S_valid = X_valid
    #---------------

    for hIdx in range(l_enc):
        #---------------
        end          = 0
        best_loss    = Infinity
        patience_cnt = 0
        #---------------

        for i in range(n_epochs[hIdx]):
            #---------------
            if end == 1:
                break
            #---------------

            #---------------
            indices = np.random.permutation(size)
            #---------------

            #---------------
            for j in range(n_batches_SAE):
                batch_s = S_train[indices[(j)*batch_size_SAE:(j+1)*batch_size_SAE]]
                
                sess.run(train_op_p[hIdx], feed_dict={S[hIdx]: batch_s, drop_prob: dropout})
                
                if (j==(n_batches_SAE-1)):
                    loss = sess.run(loss_op_p[hIdx], feed_dict={S[hIdx]: batch_s, drop_prob: dropout_0})
                    print('(Phase[{0:1d}],Epoch[{1:1d}],Batch[{2:3d}]): Minibatch Loss={3:.4f}'.format(hIdx, i, j, loss))
            #---------------

            #---------------
            # tr_loss = sess.run(loss_op_p[hIdx], feed_dict={S[hIdx]: S_train, drop_prob: dropout_0})
            # ts_loss = sess.run(loss_op_p[hIdx], feed_dict={S[hIdx]: S_valid,  drop_prob: dropout_0})
            # print('(Phase[{0:1d}],Epoch[{1:1d}]): Training Loss={2:.4f}, Validation Loss={3:.4f}'.format(hIdx, i, tr_loss, ts_loss))
            #---------------

            # Early Stopping
            #---------------
            # if (ts_loss < best_loss):
            #     best_loss    = ts_loss
            #     patience_cnt = 0
            #     print("(Phase[{0:1d}]): The Best Loss is: {1:.4f}".format(hIdx, best_loss))
            # else:
            #     patience_cnt += 1

            # if (patience_cnt >= n_patience[hIdx]):
            #     end = 1
            #     print("(Phase[{0:1d}]): Early Stopped!".format(hIdx))
            #---------------

        #---------------
        S_train = sess.run(hlayer[hIdx], feed_dict={S[hIdx]: S_train, drop_prob: dropout_0})
        # S_valid = sess.run(hlayer[hIdx], feed_dict={S[hIdx]: S_valid, drop_prob: dropout_0})
        #---------------

        # [Save the model]
        #-------------------------
        save_path = saver.save(sess, MODEL_PATH_S)
        print("Model saved in path: %s" % save_path)
        #-------------------------
    #-------------------------

    # [Restore the model]
    #-------------------------
    # saver.restore(sess, '/content/drive/MyDrive/OUTPUT_c/model_Stack.ckpt')
    # print("Model restored.")
    #-------------------------

    # [PART II]
    #-------------------------
    #---------------
    S_train = X_train.astype('float32')/255
    #---------------

    for i in range(n_epochs[0]):
        #---------------
        indices = np.random.permutation(size)
        #---------------

        #---------------
        for j in range(n_batches_SAE):
            batch_s = S_train[indices[(j)*batch_size_SAE:(j+1)*batch_size_SAE]]

            sess.run(train_op_stack, feed_dict={S[0]: batch_s, drop_prob: dropout})

            # if (j==(n_batches_SAE-1)):
            #     loss = sess.run(loss_op_stack, feed_dict={S[0]: batch_s, drop_prob: dropout_0})
            #     print('(Phase[{0:1d}],Epoch[{1:1d}],Batch[{2:3d}]): Minibatch Loss={3:.4f}'.format(-1, i, j, loss))
        #---------------

        #---------------
        tr_loss = sess.run(loss_op_stack, feed_dict={S[0]: S_train[0:1000], drop_prob: dropout_0})
        print('(Phase[{0:1d}],Epoch[{1:1d}]): Training Loss={2:.4f}'.format(-1, i, tr_loss))
        #---------------
    #-------------------------

    S_train = []

    # [Save the model]
    #-------------------------
    save_path = saver.save(sess, MODEL_PATH_A)
    print("Model saved in path: %s" % save_path)
    #-------------------------

    # [Restore the model]
    #-------------------------
    # saver.restore(sess, '/content/drive/MyDrive/OUTPUT_c/model.ckpt')
    # print("Model restored.")
    #-------------------------

    # [PART III]
    #-------------------------
    sess.run(learning_rate)

    # Prepare batches
    nextBatch_gen = nextBatch(batch_size, X_train, y_train)

    #---------------
    end          = 0
    best_acc     = 0
    train_acc    = 0
    patience_cnt = 0
    list_train   = []
    list_valid   = []
    #---------------

    for i in range(n_epochs[l_enc]):
        #---------------
        if end == 1:
            break
        #---------------

        #---------------
        for j in range(n_batches):
            batch_x, batch_y = next(nextBatch_gen)

            sess.run(train_op_class, feed_dict={S[0]: batch_x.astype('float32')/255, Y: batch_y, drop_prob: dropout})

            train_acc += sess.run(accuracy, feed_dict={S[0]: batch_x.astype('float32')/255, Y: batch_y, drop_prob: dropout_0})

            # if ((j%20)==0):
            #     loss, acc = sess.run([loss_op_class, accuracy], feed_dict={S[0]: batch_x.astype('float32')/255, Y: batch_y, drop_prob: dropout_0})
            #     print('(Epoch[{0:1d}],Batch[{1:3d}]): Minibatch Loss={2:.4f}, Training Accuracy={3:.3f}'.format(i, j, 0, acc))
        #---------------

        #---------------
        clr, loss, acc = sess.run([learning_rate, loss_op_class, accuracy], feed_dict={S[0]: batch_x.astype('float32')/255, Y: batch_y, drop_prob: dropout_0})
        train_acc     /= (n_batches)
        valid_acc      = sess.run(accuracy, feed_dict={S[0]: X_valid.astype('float32')/255, Y: y_valid, drop_prob: dropout_0})
        print("Step " + str(i) + ", Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Validation Accuracy= {:.4f}, Learning Rate= {:.4f}".format(loss, train_acc, valid_acc, clr))
        #---------------

        #---------------
        # Store training & validation accuracy
        list_train.append(train_acc)
        list_valid.append(valid_acc)

        # Reset training accuracy
        train_acc = 0
        #---------------

        # Save the model & Early Stopping
        #---------------
        if (valid_acc > best_acc):
            save_path    = saver.save(sess, MODEL_PATH)
            best_acc     = valid_acc
            patience_cnt = 0
            print("The Best Accuracy is: {0:.4f}".format(best_acc))
            print("Model saved in path: %s" % save_path)
        else:
            patience_cnt += 1

        if (patience_cnt >= n_patience[l_enc]):
            end = 1
            print("Classification: Early Stopped!")
        #---------------
    #-------------------------

    print('Optimization Finished!')
#==================================================


# Testing:
#==================================================
#---------------
sess   = tf.Session()    
y_pred = []
#---------------

with tf.Session() as sess:
    # Restore the model
    #---------------
    saver.restore(sess, save_path)
    print("Model restored.")
    #---------------

    #---------------
    y_valid_pred = sess.run(prediction, feed_dict={S[0]: X_valid.astype('float32')/255, drop_prob: dropout_0})
    valid_acc    = sess.run(accuracy,   feed_dict={S[0]: X_valid.astype('float32')/255, Y: y_valid, drop_prob: dropout_0})
    print("Validation Accuracy= {:.4f}".format(valid_acc))
    #---------------

    #---------------
    y_pred   = sess.run(prediction, feed_dict={S[0]: X_test.astype('float32')/255, drop_prob: dropout_0})
    test_acc = sess.run(accuracy,   feed_dict={S[0]: X_test.astype('float32')/255, Y: y_test, drop_prob: dropout_0})
    print("Testing Accuracy= {:.4f}".format(test_acc))
    #---------------
#==================================================


# Results:
#==================================================
# [Plot confusion matrix]
#-------------------------
#---------------
y_p = y_pred.argmax(axis = -1)
y_t = y_test.argmax(axis = -1)
#---------------

#---------------
class_names = ['lantern', 'non-lantern']
cnf_matrix  = confusion_matrix(y_t, y_p)
np.set_printoptions(precision=2)
#---------------

# Plot non-normalized confusion matrix
#---------------
plt.figure()
plotConfusionMatrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
plt.savefig(FIG_PATH)
#---------------

# Plot normalized confusion matrix
#---------------
plt.figure()
plotConfusionMatrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
plt.savefig(FIG_PATH_N)
#---------------
#-------------------------

# [Print the results]
#-------------------------
print('accuracy    = {:.5f}'.format(accuracy_score(y_t, y_p)))

print('prcision_0  = {:.5f}'.format(precision_score(y_t, y_p, pos_label=0, average='binary')))
print('prcision_1  = {:.5f}'.format(precision_score(y_t, y_p, pos_label=1, average='binary')))

print('recall      = {:.5f}'.format(recall_score(y_t, y_p,    pos_label=0, average='binary'))) 
print('specificity = {:.5f}'.format(recall_score(y_t, y_p,    pos_label=1, average='binary'))) 

print('f1-scroe_0  = {:.5f}'.format(f1_score(y_t, y_p,        pos_label=0, average='binary')))
print('f1-scroe_1  = {:.5f}'.format(f1_score(y_t, y_p,        pos_label=1, average='binary')))

print('MCC         = {:.5f}'.format(matthews_corrcoef(y_t, y_p)))
print('Kappa       = {:.5f}'.format(cohen_kappa_score(y_t, y_p)))
#-------------------------

#-------------------------
np.save('/content/drive/MyDrive/OUTPUT_c/list_train.npy', list_train)
np.save('/content/drive/MyDrive/OUTPUT_c/list_valid.npy', list_valid)
#-------------------------
#==================================================
