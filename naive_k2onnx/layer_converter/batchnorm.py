#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_tensor, make_node
from onnx import TensorProto
from onnx import numpy_helper
from . import Layer
from ..utils import np2tensor


class BatchNormalization(Layer):
    def convert(self):
        super().convert()
        assert len(self.input_full_names) == 1
        config = self.config.copy()

        # ToDo: ListWrapper
        #assert config['axis'] == -1, 'onnx batchnorm is only implemented for channel axis (1)'

        # load weights from keras layer
        scale, b, mean, var = self.layer.get_weights()
        # add initializers to graph for holding w and b
        label_scale  = self.get_label('scale')
        scale_tensor = np2tensor(scale, label_scale)
        label_b      = self.get_label('bias')
        b_tensor     = np2tensor(b, label_b)
        label_mean   = self.get_label('mean')
        mean_tensor  = np2tensor(mean, label_mean)
        label_var    = self.get_label('var')
        var_tensor   = np2tensor(var, label_var)
        self.onnx_initializers.extend([scale_tensor, b_tensor, mean_tensor, var_tensor])

        bn_node = make_node(
            'BatchNormalization',
            inputs   = [self.input_full_names[0], label_scale,
                        label_b, label_mean, label_var],
            outputs  = self.output_full_names,
            epsilon  = config['epsilon'],
            momentum = config['momentum'],
        )

        self.onnx_nodes.extend([bn_node])
