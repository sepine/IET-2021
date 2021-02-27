# !/usr/bin/env python
# encoding: utf-8

from sklearn.decomposition import KernelPCA


"""
@Description :
@Time : 2020/12/30 15:41
@Author : Kunsong Zhao
"""


class KernelPCAEmbedding(object):
    def __init__(self, n_component=14):
        self.n_component = n_component

    def kernel_rbf(self, train, test):
        rbf = KernelPCA(n_components=self.n_component, kernel='rbf')
        train_trans = rbf.fit_transform(train)
        test_trans = rbf.transform(test)
        return train_trans, test_trans

    def mapping(self, train, test):
        return self.kernel_rbf(train, test)
