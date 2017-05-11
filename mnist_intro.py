'''
this script is to practice the "mnist" tutorial for TensorFlow
this script uses a softmax regression
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#softmax equation
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

#cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

#gradient descent
lr = 0.5
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

epochs = 1000

for _ in range(epochs):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#check accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("The accuracy of this Gradient Descent classifier was", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))








