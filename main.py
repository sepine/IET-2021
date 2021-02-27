# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/3/8 19:45 
@Author : Kunsong Zhao
@Version：1.0
"""

__author__ = 'kszhao'

import json

from param import Config
from warnings import filterwarnings
from data_loader import DataLoader
from model import *
from evaluator import Evaluator

filterwarnings('ignore')

STEPS = 50

config = Config()

ea_measure_names = ['EA_Precision', 'EA_Recall', 'EA_F2']

algo_names = ['KPIDL', 'IDL']

dataset_names = [
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


columns = ['fix', 'ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt',
           'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'contains_bug']


def main():
    s = time.time()
    dl = DataLoader()
    print(time.time() - s)

    for project in dataset_names:

        result = {project: {i: {algo: {measure: []
                                       for measure in ea_measure_names}
                                for algo in algo_names}
                            for i in range(STEPS)}}

        counter = 0

        while counter != STEPS:

            print('This is the ' + str(counter + 1) + 'times ******************')

            x_train, x_test, x_train_scaled, x_test_scaled, y_train, y_test = dl.build_data(project)

            x_original_train, x_original_test = x_train, x_test

            pos_count = DataLoader.get_positive_count(y_train)
            all_count = len(y_train)

            print('IDL-based methods ******')
            for algo in algo_names:
                y_pred, y = eval(algo)(x_train_scaled, x_test_scaled,
                                       y_train, y_test,
                                       pos_count, all_count, config)

                result = Evaluator.ea_evaluate(x_original_test, y_test, y_pred, columns, ea_measure_names, result,
                                               project, counter, algo)

            counter += 1

            # reverse
            reverse_x_train = x_test_scaled
            reverse_y_train = y_test
            reverse_x_test = x_train_scaled
            reverse_y_test = y_train

            reverse_x_original_test = x_original_train

            pos_count = DataLoader.get_positive_count(reverse_y_train)
            all_count = len(reverse_y_train)

            for algo in algo_names:
                y_pred, y = eval(algo)(reverse_x_train, reverse_x_test,
                                       reverse_y_train, reverse_y_test,
                                       pos_count, all_count, config)

                result = Evaluator.ea_evaluate(reverse_x_original_test, reverse_y_test, y_pred, columns,
                                               ea_measure_names,
                                               result, project, counter, algo)

            counter += 1

        with open('./results/algo--final--' + str(STEPS) + '--' + project + '.json', 'w') as f:
            json.dump(result, f)


if __name__ == '__main__':
    main()








