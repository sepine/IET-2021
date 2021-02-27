# -*- coding: utf-8 -*-
"""
@Description :
@Time : 2020/4/11 12:11 
@Author : Kunsong Zhao
@Version：1.0
"""

__author__ = 'kszhao'


class ParamsDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v
        return super().__init__()

    def __key(self, key):
        return "" if key is None else key.lower()

    def __str__(self):
        import json
        return json.dumps(self)

    def __setattr__(self, key, value):
        self[self.__key(key)] = value

    def __getattr__(self, key):
        return self.get(self.__key(key))

    def __getitem__(self, key):
        return super().get(self.__key(key))

    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)


class Config(ParamsDict):

    def __init__(self):
        self.trainable = True
        self.checkpoint_dir = 'checkpoint/'
        self.model_name = 'model.ckpt'
        self.batch_size = 16
        self.hidden_layer = 32
        self.classes = 1
        self.epochs = 2000
        self.save_epoch = 200
        self.print_epoch = 500
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.99
        self.regularization_rate = 0.0001
        self.moving_average_decay = 0.99
