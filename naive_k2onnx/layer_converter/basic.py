#!/usr/bin/env python
# -*- coding: utf-8 -*-

from onnx.helper import make_tensor_value_info, make_node
from onnx import TensorProto
from . import Layer


class Input(Layer):
    """
    keras model inputs and outputs are processed by the converter class itself.
    this is only a dummy class which yields no nodes.
    """
    def convert(self):
        super().convert()
        assert len(self.output_full_names) == 1
        return
