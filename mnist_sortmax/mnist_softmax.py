# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data/', 'Directory for storing data')
flags.DEFINE_string('summary_dir', '../../tensorboard_logdir/', 'Directory for storing summary')

if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
tf.gfile.MakeDirs(FLAGS.summary_dir)


mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='bias')
y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')

image_shaped = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', image_shaped, 10)

mean_w = tf.reduce_mean(W)
mean_b = tf.reduce_mean(b)
tf.summary.scalar('mean/w', mean_w)
tf.summary.scalar('mean/b', mean_b)

tf.summary.histogram('bias', b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='corss-entropy')
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.summary.scalar('cross-entropy', cross_entropy)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

# Train
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary, _ = sess.run([merged, train_step], 
            feed_dict={x: batch_xs, y_: batch_ys})
    
    train_writer.add_summary(summary, i)


train_writer.close()

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))



