# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 23:00:23 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#import time

# Parameters
BATCHSIZE = 100 # 100
SEQ_LENGTH = 784
HIDDEN_UNITS = 32
OUT_CELLS = 100
NCLASSES = 10
CHKPOINT = "C:/Users/Chris/SkyDrive/MSc/Advanced_Topics/Assignment2/LSTM32_params.ckpt"
#CHKPOINT2 = "C:/Users/Chris/SkyDrive/MSc/Advanced_Topics/Assignment2/LSTM32_params_VS2.ckpt"

def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#train_images = np.expand_dims(binarize(mnist.train.images), axis=2)
test_images = np.expand_dims(binarize(mnist.test.images), axis=2)
testsize = test_images.shape[0]
testbatches = testsize//BATCHSIZE

# Clear graph and set up placeholders for data
tf.reset_default_graph()
#x = tf.placeholder(tf.float32, [None, 784, 1])
#ytrue = tf.placeholder(tf.float32, [None, 10])
x = tf.placeholder(tf.float32, [BATCHSIZE, SEQ_LENGTH, 1])
ytrue = tf.placeholder(tf.float32, [BATCHSIZE, NCLASSES])

# Define RNN model
cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS, state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# Set up variables
W1 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUT_CELLS],stddev=0.1), name='W1')
b1 = tf.Variable(tf.constant(0.1, shape=[OUT_CELLS]), name='b1')
W2 = tf.Variable(tf.truncated_normal([OUT_CELLS, NCLASSES],stddev=0.1), name='W2')
b2 = tf.Variable(tf.constant(0.1, shape=[NCLASSES]), name='b2')

out1 = tf.nn.relu(tf.matmul(last, W1) + b1)
y = tf.matmul(out1, W2) + b2

# Define loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, ytrue))

# Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ytrue,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to restore variables.
#saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
saver = tf.train.Saver()

# Initialise the variables
#init = tf.global_variables_initializer()

with tf.Session() as sess:
#    sess.run(init)
    saver.restore(sess, CHKPOINT)

    # Run NUMITER minibatch updates
    test_acc = 0
    test_cross = 0
    for j in range(testbatches):
        test_acc_new, test_cross_new = sess.run([accuracy, cross_entropy],
          feed_dict={x: test_images[j*BATCHSIZE:(j+1)*BATCHSIZE],
                     ytrue: mnist.test.labels[j*BATCHSIZE:(j+1)*BATCHSIZE]})
        test_acc += test_acc_new
        test_cross += test_cross_new
    test_acc = test_acc/testbatches
    test_cross = test_cross/testbatches

    print('Test accuracy:',  "{:.1%}".format(test_acc))
#    print('Train accuracy:',  "{:.1%}".format(train_acc),
#                  ' Test accuracy:', "{:.1%}".format(test_accuracy[snap]))        

#    save_path = saver.save(sess, CHKPOINT2)
#    print("Model saved in file: %s" % save_path)  

            
    final_state_restored = sess.run(state, feed_dict={x: test_images[0:BATCHSIZE], 
                                                    ytrue: mnist.test.labels[0:BATCHSIZE]})
    W1_r, b1_r, W2_r, b2_r = sess.run([W1,b1,W2,b2], feed_dict={x: test_images[0:BATCHSIZE], 
                                                    ytrue: mnist.test.labels[0:BATCHSIZE]})
