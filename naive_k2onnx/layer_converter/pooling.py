#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_tensor, make_node
from onnx import TensorProto
from onnx import numpy_helper
from . import Layer
from ..utils import np2tensor


class AveragePooling(Layer):
    def convert(self):
        super().convert()
        assert len(self.input_full_names) == 1
        config = self.config.copy()

        if config['padding'] == 'same': # ToDo
            auto_pad = 'SAME_UPPER'
        elif config['padding'] == 'valid':
            auto_pad = 'VALID'
        else:
            raise NotImplementedError(f'padding "{config["padding"]}" not implemented')

        avgpooling_node = make_node(
            'AveragePool',
            inputs   = self.input_full_names,
            outputs  = self.output_full_names,
            auto_pad = auto_pad,
            kernel_shape = config['pool_size'],
            strides  = config['strides'],
        )

        self.onnx_nodes.extend([avgpooling_node])
