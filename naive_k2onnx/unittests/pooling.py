#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling3D
from . import TestTemplate, get_sample_data


class TestAveragePooling3D(TestTemplate):

    def run_experiment(self, config, input_shape=(1,96,102,99,10)):
        K.clear_session()
        input_tensor  = Input(name='input_0', shape=input_shape[1:])
        output_tensor = AveragePooling3D(**config)(input_tensor)
        keras_model   = Model(input_tensor, output_tensor)
        keras_model.compile(optimizer="Adam", loss="mse")
        data_c_last, data_c_first = get_sample_data(input_shape)
        data_c_last  = {'input_0': data_c_last}
        data_c_first = {'input_0': data_c_first}
        super().run_experiment(keras_model, data_c_last, data_c_first)


    def test_avg_pooling_3d_padding_same(self):
        config = {'pool_size': [2, 2, 3],
                  'padding': 'same'}
        self.run_experiment(config)

    def test_avg_pooling_3d_padding_valid(self):
        config = {'pool_size': [2, 2, 3],
                  'padding': 'valid'}
        self.run_experiment(config)