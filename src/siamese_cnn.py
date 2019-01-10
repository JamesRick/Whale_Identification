import tensorflow as tf
import numpy as np
import pandas as pd
import pdb


def siamese_cnn_layers(input, reuse=False):
    with tf.name_scope('siamese_cnn_layers'):
        with tf.variable_scope('conv_layer_1') as scope:
            layer = tf.layers.conv2d(inputs=input, filters=64, kernel_size=[10, 10],
                padding='SAME', activation_fn=tf.nn.relu, kernel_regularizer=tf.layers.l2_regularizer,
                name=scope, reuse=reuse)
            pool = tf.layers.max_pool2d(inputs=layer, pool_size=[2, 2], strides=2, padding='SAME')

        with tf.variable_scope('conv_layer_2') as scope:
            layer = tf.layers.conv2d(inputs=pool, filters=128, kernel_size=[7, 7],
                padding='SAME', activation_fn=tf.nn.relu, kernel_regularizer=tf.layers.l2_regularizer,
                name=scope, reuse=reuse)
            pool = tf.layers.max_pool2d(inputs=layer, pool_size=[2, 2], strides=2, padding='SAME')

        with tf.variable_scope('conv_layer_3') as scope:
            layer = tf.layers.conv2d(inputs=pool, filters=128, kernel_size=[4, 4],
                padding='SAME', activation_fn=tf.nn.relu, kernel_regularizer=tf.layers.l2_regularizer,
                name=scope, reuse=reuse)
            pool = tf.layers.max_pool2d(inputs=layer, pool_size=[2, 2], strides=2, padding='SAME')

        with tf.variable_scope('conv_layer_4') as scope:
            layer = tf.layers.conv2d(inputs=pool, filters=256, kernel_size=[4, 4],
                padding='SAME', activation_fn=tf.nn.relu, kernel_regularizer=tf.layers.l2_regularizer,
                name=scope, reuse=reuse)
            dense_input = tf.layers.max_pool2d(inputs=layer, pool_size=[2, 2], strides=2, padding='SAME')

    return dense_input

def siamese_dense_layer(twin_1, twin_2):
    twin_1_dense_layer = tf.layers.dense(inputs=twin_1, units=4096,
        activation=tf.nn.relu, use_bias=True,
        kernel_regularizer=tf.layers.l2_regularizer, name='dense_1',
        reuse=False)

    twin_2_dense_layer = tf.layers.dense(inputs=twin_2, units=4096,
        activation=tf.nn.sigmoid, use_bias=True,
        kernel_regularizer=tf.layers.l2_regularizer, name='dense_1',
        reuse=True)

    l1_distance = tf.abs(tf.subtract(twin_1_dense_layer_1, twin_2_dense_layer_1))

    dense_output = tf.layers.dense(inputs=l1_distance, units=1,
        activation=tf.nn.sigmoid, use_bias=True,
        kernel_regularizer=tf.layers.l2_regularizer, name='dense_output')

    return dense_output

def short_siamese_dense_layer(twin_1, twin_2):
    twin_1 = tf.layers.flatten(twin_1)
    twin_2 = tf.layers.flatten(twin_2)
    l1_distance = tf.abs(tf.subtract(twin_1, twin_2))
    l1_distance_sigmoid = tf.sigmoid(l1_distance)

    dense_output = tf.layers.dense(inputs=l1_distance, units=1,
        activation=tf.nn.sigmoid, use_bias=True,
        kernel_regularizer=tf.layers.l2_regularizer, name='dense_output')

    return dense_output

def train_siamese():
    next_batch = "TEMP BATCH"
    twin_1 = tf.placeholder(tf.float32, shape=(700, 1050), name='twin_1')
    twin_2 = tf.placeholder(tf.float32, shape=(700, 1050), name='twin_2')
    label = tf.placeholder(tf.int32, shape=[None, 1], name='label')
    float_label = tf.to_float(label)

    twin_1_output = siamese_cnn_layers(twin_1, reuse=False)
    twin_2_output = siamese_cnn_layers(twin_2, reuse=True)

    dense_output = siamese_dense_layer(twin_1_output, twin_2_output)

    loss = tf.losses.sparse_softmax_cross_entropy(dense_output, labels)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss

    step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(learning_rate=0.01,
        momentum=0.99, use_nesterov=True).minimize(loss, global_step=step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
            batch_twin_1, batch_twin_2, batch_labels = "TEMP ASSIGNMENT"
            _, l = sess.run([train_step, loss], feed_dict={twin_1:batch_twin_1, twin_2:batch_twin_2, label:batch_labels})

            print("Loss at epoch " + str(i) + ":" + str(l))
