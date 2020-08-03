#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_tensor, make_node
from onnx import TensorProto
from onnx import numpy_helper
from . import Layer
from ..utils import np2tensor, to_onnx_axis


class Concatenate(Layer):
    def convert(self):
        super().convert()
        config = self.config.copy()

        axis = to_onnx_axis(config['axis'])

        concat_node = make_node(
            'Concat',
            inputs  = self.input_full_names,
            outputs = self.output_full_names,
            axis    = axis,
        )

        self.onnx_nodes.extend([concat_node])
