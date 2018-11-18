


from __future__ import print_function
import tensorflow as tf

import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
import gzip
import pandas as pd

save_file='./model.ckpt'

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)


args = parser.parse_args()
#print(tf.reduce_sum([[1,2],[3,4]],reduction_indices=[0,1]))

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1, phase_train.name: True})
    
    error=tf.reduce_sum((abs(y_pre-v_ys)))

    
    result1 = sess.run(error, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})

    return result1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')

def leakyrelu(x, alpha=0.3, max_value=None):  #alpha need set
    '''ReLU.

    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def full_batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed



num_bins=3
xs = tf.placeholder(tf.float32, [None, num_bins])   # 28x28
ys = tf.placeholder(tf.float32, [None, 1])  #num_p add 1 om
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

W_fc1 = weight_variable([num_bins, 104])
b_fc1 = bias_variable([104])
W_fc2 = weight_variable([104, 52])
b_fc2 = bias_variable([52])
W_fc3 = weight_variable([52, 1])
b_fc3 = bias_variable([1])

saver = tf.train.Saver()  #define saver of the check point

# h_fc1 = tf.nn.tanh(full_batch_norm(tf.matmul(tf.reshape(xs,[-1,num_bins]), W_fc1) + b_fc1, 104, phase_train))
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# h_fc2 = tf.nn.tanh(full_batch_norm(tf.matmul(h_fc1_drop , W_fc2) + b_fc2, 52, phase_train))
h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(xs,[-1,num_bins]), W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
h_fc2 = tf.nn.relu((tf.matmul(h_fc1_drop , W_fc2) + b_fc2))

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
prediction = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
loss = tf.reduce_mean(tf.reduce_sum(np.square(ys - prediction),
                                        reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)



x_file = open("x.p",'r')
y_file = open("y.p",'r')
X=np.array(pickle.load(x_file))
Y=np.array(pickle.load(y_file))
Y=Y.reshape(len(Y),1)
# print(np.shape(Y))

num_test=1000
num_train=len(Y)-num_test

X_train=X[:num_train, :]
y_train=Y[:num_train]
X_test=X[num_train:, :]
y_test=Y[num_train:]
index_test=[i for i in range(num_test)]

batch_size=10
loss_set=[]
step_set=[]
if args.train:
    for i in range(1000):
        print(i)
        pre,_, train_loss=sess.run([prediction,train_step,loss], feed_dict={xs: X_train[batch_size*i:batch_size*i+batch_size], ys:y_train[batch_size*i:batch_size*i+batch_size], \
        keep_prob: 0.9, lr:0.00001, phase_train.name: True})
        print(pre)
        loss_set.append(train_loss)
        step_set.append(i)
    
    plt.plot(step_set,loss_set)

    saver.save(sess, save_file)
    #plt.ylim(0,50)
    plt.savefig('com.png')
    plt.show()
if args.test:
    predict_d=sess.run(prediction, feed_dict={xs: X_test, ys:y_test, keep_prob: 1, lr:0.00001, phase_train.name: False})
    predict_d=np.reshape(np.array(predict_d),-1)
    index_set=[i for i in range (len(predict_d))]

    # mse = (np.square(predict_d - y_test)*(1./np.square(y_test))).mean()
    me = (np.abs(predict_d - y_test)).mean()
    print(np.shape(y_test))
    print(me)
    plt.scatter(index_set, predict_d,s=8)
    plt.scatter(index_set, y_test,s=4)
    # print(predict_d)
    plt.show()

   
