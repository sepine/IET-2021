# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/3/8 19:42 
@Author : Kunsong Zhao
@Version：1.0
"""
import os
import time

__author__ = 'kszhao'

import numpy as np
import tensorflow as tf

from kernel_pca import KernelPCAEmbedding
from loss import Loss

FEATURES_NUM = 14


class Model:

    def __init__(self, input_shape, hidden_shape, classes, positive_num, all_num):

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.classes = classes

        self.positive_num = positive_num
        self.all_num = all_num

    def inference(self, input_tensor, regularizer):
        """
        前向传播过程
        :param input_tensor:
        :param regularizer:
        :return:
        """
        with tf.variable_scope('layer1'):
            weights = get_weight_variable([self.input_shape, self.hidden_shape], regularizer)
            biases = tf.get_variable(
                'biases', [self.hidden_shape],
                initializer=tf.constant_initializer(0.0))

            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

        with tf.variable_scope('layer2'):
            weights = get_weight_variable([self.hidden_shape, self.hidden_shape], regularizer)
            biases = tf.get_variable(
                'biases', [self.hidden_shape],
                initializer=tf.constant_initializer(0.0))

            layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

        with tf.variable_scope('layer3'):
            weights = get_weight_variable([self.hidden_shape, self.classes], regularizer)
            biases = tf.get_variable(
                'biases', [self.classes],
                initializer=tf.constant_initializer(0.0))

            layer3 = tf.nn.sigmoid(tf.matmul(layer2, weights) + biases)

        return layer3

    def train_batch(self, x_train, y_train, config):
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, self.input_shape], name='features')
        y_ = tf.placeholder(tf.float32, [None, self.classes], name='labels')

        regularizer = tf.contrib.layers.l2_regularizer(config.regularization_rate)

        y = self.inference(x, regularizer)
        global_step = tf.Variable(0, trainable=False)

        varibale_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
        varibale_averages_op = varibale_averages.apply(tf.trainable_variables())
        loss = Loss(y_, y, self.positive_num, self.all_num).build()
        loss_total = loss + tf.add_n(tf.get_collection('losses'))

        learning_rate = tf.train.exponential_decay(
            config.learning_rate, global_step, self.all_num / config.batch_size, config.learning_rate_decay)

        train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_total, global_step=global_step)

        with tf.control_dependencies([train_step, varibale_averages_op]):
            train_op = tf.no_op(name='train')

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            count = 0

            start_time = time.time()

            print("Training...")

            loss_last = 10

            batch_idxs = len(x_train) // config.batch_size

            for epoch in range(config.epochs):

                error_total = 0

                np.random.seed(epoch)
                np.random.shuffle(x_train)
                np.random.seed(epoch)
                np.random.shuffle(y_train)

                for idx in range(0, batch_idxs):
                    batch_x = x_train[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_y = y_train[idx * config.batch_size: (idx + 1) * config.batch_size]

                    count += 1
                    _, error, step = sess.run([train_op, loss_total, global_step], feed_dict={
                        x: batch_x, y_: batch_y})

                    if count % config.print_epoch == 0:
                        print("Epoch:[%2d], step:[%2d], time:[%4.2f秒], loss: [%.16f]"
                              % ((epoch + 1), count, time.time() - start_time, error))

                    error_total += error

                    if (count % config.save_epoch == 0) and (error_total / batch_idxs < loss_last):

                        saver.save(sess, os.path.join(config.checkpoint_dir, config.model_name),
                                   global_step=global_step)
                    else:
                        pass

                error_total = error_total / batch_idxs

                if count % config.print_epoch == 0:
                    print("Epoch: [%2d], total_loss: [%.12f]" %
                          ((epoch + 1), error_total))

                if error_total < 0.05:
                    break

                loss_last = error_total

    def test(self, x_test, y_test, config):

        with tf.Graph().as_default() as g:

            x = tf.placeholder(tf.float32, [None, self.input_shape], name='features')
            y_ = tf.placeholder(tf.float32, [None, self.classes], name='labels')

            print('Test..........')

            y = self.inference(x, None)

            variable_average = tf.train.ExponentialMovingAverage(config.moving_average_decay)
            variable_to_restore = variable_average.variables_to_restore()

            saver = tf.train.Saver(variable_to_restore)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:

                    saver.restore(sess, ckpt.model_checkpoint_path)

                pred = sess.run(y, feed_dict={x: x_test,
                                              y_: y_test})

            threshold = 0.5

            res = (pred > threshold).astype(int)

            return res, y_test


def get_weight_variable(shape, regularizer):

    weights = tf.get_variable(
        'weights', shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))

    return weights


def KPIDL(x_train_scaled, x_test_scaled, y_train, y_test,
             pos_count, all_count, config):

    k_pac_emb = KernelPCAEmbedding()
    x_train_scaled, x_test_scaled = k_pac_emb.mapping(
        x_train_scaled, x_test_scaled)

    net = Model(FEATURES_NUM,
                hidden_shape=config.hidden_layer,
                classes=config.classes,
                positive_num=pos_count,
                all_num=all_count)

    net.train_batch(x_train_scaled, y_train, config)

    y_pred, y = net.test(x_test_scaled, y_test, config)

    return y_pred, y


def IDL(x_train_scaled, x_test_scaled, y_train, y_test,
        pos_count, all_count, config):

    net = Model(FEATURES_NUM,
                hidden_shape=config.hidden_layer,
                classes=config.classes,
                positive_num=pos_count,
                all_num=all_count)

    net.train_batch(x_train_scaled, y_train, config)

    y_pred, y = net.test(x_test_scaled, y_test, config)

    return y_pred, y
