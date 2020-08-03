#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization
from . import TestTemplate, get_sample_data


class TestBatchnorm(TestTemplate):

    def run_experiment(self, config, input_shape=(1,96,102,99,1)):
        K.clear_session()
        # use random initialization to better find bugs
        config['beta_initializer']            = 'random_uniform'
        config['gamma_initializer']           = 'random_uniform'
        config['moving_mean_initializer']     = 'random_uniform'
        config['moving_variance_initializer'] = 'random_uniform'
        input_tensor  = Input(name='input_0', shape=input_shape[1:])
        output_tensor = BatchNormalization(**config)(input_tensor)
        keras_model   = Model(input_tensor, output_tensor)
        keras_model.compile(optimizer="Adam", loss="mse")
        data_c_last, data_c_first = get_sample_data(input_shape)
        data_c_last  = {'input_0': data_c_last}
        data_c_first = {'input_0': data_c_first}
        super().run_experiment(keras_model, data_c_last, data_c_first)


    def test_batchnorm_center_scale(self):
        config = {'center': True, 
                  'scale':  True}
        self.run_experiment(config)
    
    
    def test_batchnorm_center(self):
        config = {'center': True, 
                  'scale':  False}
        self.run_experiment(config)
    
    def test_batchnorm_scale(self):
        config = {'center': False, 
                  'scale':  True}
        self.run_experiment(config)
    
    def test_batchnorm_without_center_or_scale(self):
        config = {'center': False, 
                  'scale':  False}
        self.run_experiment(config)