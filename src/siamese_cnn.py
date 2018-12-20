import tensorflow as tf
import numpy as np
import pandas as pd
import pdb

graph = tf.Graph()

with graph.as_default():

    # Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Model
    def model(data):
        conv1 = tf.layers.conv2d(
            inputs=data,
            filters=64,
            kernel_size=[10,10]
            padding="same",
            activation=tf.nn.relu)
    # Training Computation

    # Optimizer
    optimizer = tf.train.Momentum(learning_rate=0.5, momentum=0.01).minimize(loss)
    # Predictions
