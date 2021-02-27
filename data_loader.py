# -*- coding: utf-8 -*-
"""
@Description : Load original dataset
@Time : 2020/3/8 10:19 
@Author : Kunsong Zhao
@Version：1.0
"""
import time
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__author__ = 'kszhao'


class DataLoader:

    def __init__(self):
        self.base = "./datasets/"
        self.dataset_names = [
                              'aFall',
                              'Alfresco',
                              'androidSync',
                              'androidWalpaper',
                              'anySoftKeyboard',
                              'Apg',
                              'kiwis',
                              'owncloudandroid',
                              'Pageturner',
                              'reddit',
                              'image',
                              'lottie',
                              'ObservableScrollView',
                              'Applozic',
                              'delta_chat'
                              ]
        self.dataset = self.__load_data()

    @staticmethod
    def read_xlsx(path):
        df = pd.read_excel(path)
        df.fillna(value=0, inplace=True)
        return df

    def load_csv(self, path):
        data = []
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(np.array(row))
        data = np.array(data)
        (n, d) = data.shape
        return data, n, d

    def __load_data(self):
        tmp_dataset = OrderedDict()
        for name in self.dataset_names:
            tmp_dataset[name] = self.read_xlsx(self.base + name + '.xlsx')
        return tmp_dataset

    def build_data(self, dataset_name):
        df = self.dataset[dataset_name]
        x_train, x_test = self.__split_dataset(df)
        x_train, x_test, y_train, y_test = self.__split_train_test_label(x_train, x_test)
        x_train_scaled, x_test_scaled = self.__z_score(x_train, x_test)

        return x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test

    @staticmethod
    def __split_dataset(df, values=[1, 0], key='contains_bug'):
        """
        Split dataset
        """
        positives, negatives = (df[df[key] == v] for v in values)
        (p_train, p_test), (n_train, n_test) = map(
            lambda dataset: train_test_split(dataset, test_size=0.5,
                                             shuffle=True, random_state=None),
            (positives, negatives))

        return p_train.append(n_train), p_test.append(n_test)

    @staticmethod
    def __split_train_test_label(x_train, x_test):
        """
        Split the features and labels
        """
        x_train, x_test = x_train.loc[:].values, x_test.loc[:].values
        y_train = x_train[:, [-1]]
        x_train = x_train[:, : -1]
        y_test = x_test[:, [-1]]
        x_test = x_test[:, : -1]

        return x_train, x_test, y_train, y_test

    @staticmethod
    def __z_score(X_train, X_test):
        """
        Standard the features
        """
        scaler_1 = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_trian_scaled = scaler_1.fit_transform(X_train)
        scaler_2 = StandardScaler(copy=True, with_mean=True, with_std=True)
        X_test_scaled = scaler_2.fit_transform(X_test)

        return X_trian_scaled, X_test_scaled

    @staticmethod
    def get_positive_count(dataset):
        pos_count = list(dataset).count(1)
        return pos_count



