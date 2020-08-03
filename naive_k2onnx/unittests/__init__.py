#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logging import debug
import unittest
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input as K_Input
import onnxruntime as ort
from ..converter import Converter


class TestTemplate(unittest.TestCase):

    def run_experiment(self, keras_model, data_c_last, data_c_first,
                       rtol=1e-05, atol=1e-06):
        converter = Converter()
        onnx_model = converter.convert(keras_model)

        # calculate results tf
        r_tf = keras_model.predict(data_c_last)

        # calculate results orrt
        session = ort.InferenceSession(onnx_model.SerializeToString())
        r_rt = session.run(None, data_c_first)[0]
        r_rt = np.moveaxis(r_rt, 1, -1)

        debug('    MAE: %f' % np.mean(np.abs(r_tf - r_rt)))
        debug('    std: %f' % np.std(r_tf - r_rt))

        if not np.allclose(r_tf, r_rt, rtol=rtol, atol=atol):
            raise Exception('results diverge')


def get_sample_data(n_dim, dtype=np.float32):
    """
        n_dim = b,x,y...,c
        returns data_tf, data_pt 
    """
    data_c_last  = np.random.randn(*n_dim).astype(dtype)
    data_c_first = np.moveaxis(data_c_last, -1, 1)
    return data_c_last, data_c_first
