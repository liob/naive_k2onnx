#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, PReLU
from . import TestTemplate, get_sample_data


class TestPReLU(TestTemplate):

    def run_experiment(self, config, input_shape=(1,96,102,99,10)):
        K.clear_session()
        # use random initialization to better find bugs
        config['alpha_initializer'] = 'random_uniform'
        # onnx only supports shared color channel
        config['shared_axes'] = [1,2,3]
        input_tensor  = Input(name='input_0', shape=input_shape[1:])
        output_tensor = PReLU(**config)(input_tensor)
        keras_model   = Model(input_tensor, output_tensor)
        keras_model.compile(optimizer="Adam", loss="mse")
        data_c_last, data_c_first = get_sample_data(input_shape)
        data_c_last  = {'input_0': data_c_last}
        data_c_first = {'input_0': data_c_first}
        super().run_experiment(keras_model, data_c_last, data_c_first)


    def test_prelu(self):
        config = {}
        self.run_experiment(config)
