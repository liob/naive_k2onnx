#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx
from onnx.helper import make_tensor_value_info, make_tensor, make_node
from onnx import TensorProto
from onnx import numpy_helper
from . import Layer
from ..utils import np2tensor


class PReLU(Layer):
    def convert(self):
        super().convert()
        assert len(self.input_full_names) == 1
        assert len(self.output_full_names) == 1
        config = self.config.copy()

        alpha = self.layer.get_weights()[0]
        alpha = np.moveaxis(alpha, -1, 0) # nhwc -> nchw
        label_alpha  = self.get_label('alpha')
        alpha_tensor = np2tensor(alpha, label_alpha)
        self.onnx_initializers.extend([alpha_tensor])

        prelu_node = make_node(
            'PRelu',
            inputs   = [self.input_full_names[0], label_alpha],
            outputs  = self.output_full_names,
        )

        self.onnx_nodes.extend([prelu_node])
