'''
This file is to practice the development of a linear classifier using TensorFlow
'''

from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01 #learning rate
epochs = 1000

#training data
nums_x = np.random.rand(1,10)*10
nums_y = np.random.rand(1,10)*10
#x_inp = np.asarray(nums_x)
#y_inp = np.asarray(nums_y)
x_inp = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                    7.042,10.791,5.313,7.997,5.654,9.27,3.1])
y_inp = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                    2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n = x_inp.shape[0]

x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# model

model = tf.add(tf.multiply(x, w), b)

cost = tf.reduce_sum(tf.pow(model-y,2))/(2*n)
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        for (X, Y) in zip(x_inp,y_inp):
            sess.run(optimizer, feed_dict={x: X, y: Y})

    plt.plot(x_inp,y_inp,'ro',label='Input Data')
    plt.plot(x_inp, sess.run(w) * x_inp + sess.run(b), label='Trained line')
    plt.show()

    # testing the classifier

    test_x = np.random.rand(1,8)*10
    test_y = np.random.rand(1,8)*10

    testing_cost = sess.run(tf.reduce_sum(tf.pow(model - y, 2)) / (2 * test_x.shape[0]),
                            feed_dict={y: test_x, x: test_y})

    plt.plot(test_x, test_y, 'bo', label='Testing data')
    plt.plot(x_inp, sess.run(w) * x_inp + sess.run(b), label='Fitted line')
    plt.show()







