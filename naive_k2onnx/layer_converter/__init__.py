#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ..utils import tf_name2canonical, to_onnx_shapes


class Layer(object):
    def __init__(self, layer):
        self.layer = layer
    
    @property
    def input_full_names(self):
        inputs = self.layer.input
        if not isinstance(inputs, list):
            inputs = [inputs]
        return [tf_name2canonical(tensor.name) for tensor in inputs]

    @property
    def output_full_names(self):
        outputs = self.layer.output
        if not isinstance(outputs, list):
            outputs = [outputs]
        return [tf_name2canonical(tensor.name) for tensor in outputs]
    
    @property
    def config(self):
        return self.layer.get_config()
    
    @property
    def input_shape(self):
        assert len(self.input_shapes) == 1
        return self.input_shapes[0]
    
    @property
    def input_shapes(self):
        return to_onnx_shapes(self.layer.input_shape)
    
    @property
    def output_shape(self):
        assert len(self.output_shapes) == 1
        return self.output_shapes[0]
    
    @property
    def output_shapes(self):
        return to_onnx_shapes(self.layer.output_shape)
    
    @property
    def in_channels(self):
        return self.layer.input_shape[-1]
    
    @property
    def out_channels(self):
        return self.layer.output_shape[-1]
    
    def convert(self):
        self.onnx_nodes        = []
        self.onnx_initializers = []

    def get_label(self, name):
        return f'{self.layer.name}/{name}'


from .basic import NoOp, Input
from .conv import Conv
from .batchnorm import BatchNormalization
from .activation import PReLU
from .pooling import AveragePooling
from .merge import Concatenate
