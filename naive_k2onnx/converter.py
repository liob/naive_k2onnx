#!/usr/bin/env python
# -*- coding: utf-8 -*-

import onnx
from onnx import TensorProto
from onnx.helper import make_graph, make_model
from onnx.helper import make_tensor_value_info
import tensorflow as tf
import tensorflow.python.keras.layers as tf_layers
from .layer_converter import Input, Conv, BatchNormalization, PReLU, AveragePooling, Concatenate
from .utils import tf_name2canonical, to_onnx_shapes


class Converter(object):
    def __init__(self):
        self.converters = {}

        self.add_converter(tf.python.keras.engine.input_layer.InputLayer, Input)
        self.add_converter(tf_layers.convolutional.Conv3D, Conv)
        self.add_converter(tf_layers.normalization_v2.BatchNormalization, BatchNormalization)
        self.add_converter(tf_layers.advanced_activations.PReLU, PReLU)
        self.add_converter(tf_layers.pooling.AveragePooling3D, AveragePooling)
        self.add_converter(tf_layers.merge.Concatenate, Concatenate)

    
    def convert(self, keras_model, opset=11, model_checker=True):
        nodes, initializers = [], []
        for keras_layer in keras_model.layers:
            layer_type = type(keras_layer)
            if not layer_type in self.converters:
                raise NotImplementedError(f'Layer type ({str(layer_type)}) not implemented!')
            op = self.converters[layer_type](keras_layer)
            op.convert()
            nodes.extend(op.onnx_nodes)
            initializers.extend(op.onnx_initializers)
        
        # create model inputs
        inputs = []
        for input in keras_model.inputs:
            input_shape  = to_onnx_shapes([input.shape])[0]
            input_tensor = make_tensor_value_info(
                                              tf_name2canonical(input.name),
                                              TensorProto.FLOAT, # ToDo: use actual data type
                                              input_shape)
            inputs.append(input_tensor)

        # create model outputs
        outputs = []
        for output in keras_model.outputs:
            output_shape  = to_onnx_shapes([output.shape])[0]
            output_tensor = make_tensor_value_info(
                                              tf_name2canonical(output.name),
                                              TensorProto.FLOAT, # ToDo: use actual data type
                                              output_shape)
            outputs.append(output_tensor)
        
        # Create the graph (GraphProto)
        graph_def = make_graph(
            nodes=nodes,
            name=keras_model.name,
            inputs=inputs,
            outputs=outputs,
            initializer=initializers,
        )

        # Create the model (ModelProto)
        model_def = make_model(graph_def, producer_name='naive_k2onnx')
        model_def.opset_import[0].version = opset
        #onnx.save(model_def, '/tmp/test.onnx')
        if model_checker:
            onnx.checker.check_model(model_def)
        return model_def


    def add_converter(self, layer_type, converter):
        self.converters[layer_type] = converter
