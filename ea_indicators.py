# !/usr/bin/env python
# encoding: utf-8

"""
@Description :
@Time : 2020/9/18 19:56
@Author : Kunsong Zhao
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.metrics import auc


def get_loc_data(target_datas, target_label, columns):
    target_label = target_label.astype('int')
    target_label = np.reshape(target_label, newshape=(len(target_label), 1))
    target_datas = np.hstack((target_datas, target_label))
    df = pd.DataFrame(target_datas, columns=columns)
    df["bug"] = df.pop(df.columns[-1])
    df["loc"] = df["ld"] + df["la"]
    return df


# Move the positive instances to the front of the dataset.
def positive_first(df: DataFrame) -> DataFrame:
    return concat([df[df.pred == True], df[df.pred == False]])


def effort_aware(df: DataFrame, EAPredict: DataFrame):
    """Calculate the effort-aware performance indicators."""
    EAOptimal = concat([df[df.bug == True], df[df.bug == False]])
    EAWorst = EAOptimal.iloc[::-1]

    M = len(df)
    N = sum(df.bug)
    m = threshold_index(EAPredict['loc'], 0.2)
    n = sum(EAPredict.bug.iloc[:m])
    for k, y in enumerate(EAPredict.bug):
        if y:
            break

    y = set(vars().keys())
    EA_Precision = n / m
    EA_Recall = n / N
    EA_F1 = harmonic_mean(EA_Precision, EA_Recall)
    EA_F2 = 5 * EA_Precision * EA_Recall / np.array(4 * EA_Precision + EA_Recall + 1e-8)
    # PCI = m / M
    # IFA = k
    P_opt = norm_opt(EAPredict, EAOptimal, EAWorst)
    M = vars()
    # print("EAPredict", EAPredict)
    # print("EAOptimal", EAOptimal)
    # print("EAWorst", EAWorst)
    return {k: M[k] for k in reversed(list(M)) if k not in y}


def threshold_index(loc, percent: float) -> int:
    threshold = sum(loc) * percent
    for i, x in enumerate(loc):
        threshold -= x
        if threshold < 0:
            return i + 1


def norm_opt(*args) -> float:
    'Calculate the Alberg-diagram-based effort-aware indicator.'
    predict, optimal, worst = map(alberg_auc, args)
    return 1 - (optimal - predict) / (optimal - worst)


# df: target domain内被split分割的train data
def alberg_auc(df: DataFrame) -> float:
    'Calculate the area under curve in Alberg diagrams.'
    points = df[['loc', 'bug']].values.cumsum(axis=0)
    points = np.insert(points, 0, [0, 0], axis=0) / points[-1]
    return auc(*points.T)


def harmonic_mean(x, y, beta=1):
    beta *= beta
    # return (beta + 1) * x * y / (beta * x + y)
    return (beta + 1) * x * y / (beta * x + y + 1e-8)

