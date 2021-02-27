# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/3/7 17:00 
@Author : Kunsong Zhao
@Version：1.0
"""

__author__ = 'kszhao'

import numpy as np
import tensorflow as tf


class Loss:

    def __init__(self, y, y_pred, positive_number, all_number):
        self.positive_number = positive_number
        self.all_number = all_number
        self.lmbda = self.positive_number / self.all_number
        self.y = y
        self.y_pred = y_pred

    def build(self):

        EPS = 10e-6

        cost = tf.reduce_mean(tf.reduce_sum(
            -self.y * tf.log(tf.clip_by_value(self.y_pred, EPS, tf.reduce_max(self.y_pred))) * self.lmbda
            - ((1 - self.y) * tf.log(tf.clip_by_value(1 - self.y_pred, EPS, tf.reduce_max(self.y_pred))) * (
                        1 - self.lmbda))))

        return cost