#!/usr/bin/env python
# -*- coding: utf-8 -*-

from onnx.helper import make_tensor_value_info, make_node
from onnx import TensorProto
from . import Layer


class NoOp(Layer):
    """
    does not perform any action than mapping input to output
    """
    def convert(self):
        super().convert()
        assert len(self.input_full_names) == 1
        assert len(self.output_full_names) == 1

        noop_node = make_node(
            'Identity',
            inputs   = self.input_full_names,
            outputs  = self.output_full_names,
        )

        self.onnx_nodes.extend([noop_node])


class Input(Layer):
    """
    keras model inputs and outputs are processed by the converter class itself.
    this is only a dummy class which yields no nodes.
    """
    def convert(self):
        super().convert()
        assert len(self.output_full_names) == 1
        return
