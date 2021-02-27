# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/4/12 12:35 
@Author : Kunsong Zhao
@Version：1.0
"""

__author__ = 'kszhao'

from ea_indicators import *


class Evaluator:

    def __init__(self, y_pred, y, measure_names):
        self.y_pred = y_pred
        self.y = y
        self.measure_names = measure_names

    @staticmethod
    def ea_evaluate(x_original_test, y_test, y_pred, columns, ea_measure_names, result, project, count, algo):
        pred = y_pred

        test_df = get_loc_data(x_original_test, y_test, columns)
        test_df["pred"] = pred

        XYLuYp_sorted = test_df.sort_values('loc')
        eff_aware = effort_aware(XYLuYp_sorted, positive_first(XYLuYp_sorted))

        for measure in ea_measure_names:
            result[project][count][algo][measure] = \
                str(eff_aware[measure])

        return result


