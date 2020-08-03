#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_tensor, make_node
from onnx import TensorProto
from onnx import numpy_helper
from . import Layer
from ..utils import np2tensor


class Conv(Layer):
    def convert(self):
        super().convert()
        assert len(self.input_full_names) == 1
        assert len(self.output_full_names) == 1
        config = self.config.copy()

        # load weights from keras layer
        if config['use_bias']:
            w, b = self.layer.get_weights()
        else:
            w = self.layer.get_weights()[0]
            b = None
        # convert weights to onnx format (nhwc -> nchw)
        w = np.moveaxis(w, [-2, -1], [1, 0])
        # add initializers to graph for holding w and b
        label_w  = self.get_label('weight')
        w_tensor = np2tensor(w, label_w)
        label_b  = self.get_label('bias')
        b_tensor = np2tensor(b, label_b)
        self.onnx_initializers.extend([w_tensor, b_tensor])

        if config['padding'] == 'same': # ToDo
            auto_pad = 'SAME_UPPER'
        elif config['padding'] == 'valid':
            auto_pad = 'VALID'
        else:
            raise NotImplementedError(f'padding "{config["padding"]}" not implemented')

        conv_node = make_node(
            'Conv',
            inputs  = [self.input_full_names[0], label_w, label_b],
            outputs = self.output_full_names,
            kernel_shape = config['kernel_size'],
            auto_pad     = auto_pad,
            strides      = config['strides'],
            dilations    = config['dilation_rate'],
        )

        self.onnx_nodes.extend([conv_node])
