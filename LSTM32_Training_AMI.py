# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 23:00:23 2017

@author: Chris
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import time
import pickle

# Parameters
LEARN_RATE = 0.001 # 0.001
BATCHSIZE = 100 # 100
TRAINSIZE = 55000
EPOCHS = 5
NUMITER = EPOCHS*TRAINSIZE//BATCHSIZE
SEQ_LENGTH = 784
HIDDEN_UNITS = 32
OUT_CELLS = 100
NCLASSES = 10
SNAP_INTERVAL = 10
NUMSNAPS = NUMITER//SNAP_INTERVAL
CHKPOINT = "/LSTM32_params_new.ckpt"
#CHKPOINT = "C:/Users/Chris/SkyDrive/MSc/Advanced_Topics/Assignment2/LSTM32_params.ckpt"
PKLFILE = "LSTM32_data.pckl"
#PKLFILE = "C:/Users/Chris/SkyDrive/MSc/Advanced_Topics/Assignment2/LSTM32_data.pckl"

# Vectors for saving training and test accuracy and cross-entropy
train_accuracy = np.zeros(NUMSNAPS)
test_accuracy = np.zeros(NUMSNAPS)
train_crossent = np.zeros(NUMSNAPS)
test_crossent = np.zeros(NUMSNAPS)

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
#W = tf.Variable(tf.truncated_normal([num_hidden, int(ytrue.get_shape()[1])]))
#bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
W1 = tf.Variable(tf.truncated_normal([HIDDEN_UNITS, OUT_CELLS],stddev=0.1), name='W1')
b1 = tf.Variable(tf.constant(0.1, shape=[OUT_CELLS]), name='b1')
W2 = tf.Variable(tf.truncated_normal([OUT_CELLS, NCLASSES],stddev=0.1), name='W2')
b2 = tf.Variable(tf.constant(0.1, shape=[NCLASSES]), name='b2')

out1 = tf.nn.relu(tf.matmul(last, W1) + b1)
y = tf.matmul(out1, W2) + b2

# Define loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, ytrue))

# Define how to train the model
train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(cross_entropy)

# Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ytrue,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save all the variables.
#saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
saver = tf.train.Saver()

# Initialise the variables
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    best_accuracy = 0

    # Run NUMITER minibatch updates
    for i in range(NUMITER):
        batch_xs, batch_ys = mnist.train.next_batch(BATCHSIZE)
        batch_xsb = binarize(np.expand_dims(batch_xs, axis=2))
        sess.run(train_step, feed_dict={x: batch_xsb, ytrue: batch_ys})
        if i % SNAP_INTERVAL == 0:
            snap = i//SNAP_INTERVAL        
            train_accuracy[snap], train_crossent[snap] = sess.run([accuracy, cross_entropy],
                          feed_dict={x: batch_xsb, ytrue: batch_ys})
#            starttime = time.time()
            for j in range(testbatches):
                test_acc_new, test_cross_new = sess.run([accuracy, cross_entropy],
                  feed_dict={x: test_images[j*BATCHSIZE:(j+1)*BATCHSIZE],
                             ytrue: mnist.test.labels[j*BATCHSIZE:(j+1)*BATCHSIZE]})
                test_accuracy[snap] += test_acc_new
                test_crossent[snap] += test_cross_new
            test_accuracy[snap] = test_accuracy[snap]/testbatches
            test_crossent[snap] = test_crossent[snap]/testbatches
            print('Train accuracy:',  "{:.1%}".format(train_accuracy[snap]),
                  ' Test accuracy:', "{:.1%}".format(test_accuracy[snap]))        
            if test_accuracy[snap]>best_accuracy:
                best_accuracy = test_accuracy[snap]
                save_path = saver.save(sess, CHKPOINT)
                print("Model saved in file: %s" % save_path)                  
                
#            endtime = time.time()
#            print('Time to calc test error:', "{:.1f}".format(endtime-starttime))

    # Calculate final test accuracy
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

    print('Final Test accuracy:',  "{:.1%}".format(test_acc))
            
    # Save the variables to disk.
#    save_path = saver.save(sess, CHKPOINT)
#    print("Model saved in file: %s" % save_path)  

#    final_state = sess.run(state, feed_dict={x: batch_xsb, ytrue: batch_ys})
#    final_val = sess.run(val, feed_dict={x: batch_xsb, ytrue: batch_ys})
#    final_last = sess.run(last, feed_dict={x: batch_xsb, ytrue: batch_ys})
#    W1_tr, b1_tr, W2_tr, b2_tr = sess.run([W1,b1,W2,b2], feed_dict={x: batch_xsb, ytrue: batch_ys})

    # Calculate confusion matrix
#    y_pred, y_actu = sess.run([tf.argmax(y,1), tf.argmax(ytrue,1)],
#                               feed_dict={x: mnist.test.images, ytrue: mnist.test.labels})
#    cm = confusion_matrix(y_actu, y_pred)

    # Plot the training and test errors during training
    trainline = plt.plot(np.arange(0,NUMITER,SNAP_INTERVAL), train_accuracy.T, linewidth=2)
    testline = plt.plot(np.arange(0,NUMITER,SNAP_INTERVAL), test_accuracy.T, linewidth=2)
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.title("Training RNN with 32-unit LSTM")
    plt.show()

# Save training hisotry to disk.
pickle.dump([train_accuracy, test_accuracy, train_crossent, test_crossent], open(PKLFILE, "wb"))
