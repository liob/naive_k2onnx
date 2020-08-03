#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import onnx


def tf_name2canonical(tf_name):
    return tf_name.rsplit(':', 1)[0]


def to_onnx_shapes(shapes):
    output = []
    for shape in shapes:
        shape = np.array(shape)        # bhw...c
        shape = np.roll(shape, 1)      # cbhw...
        shape[0:2] = shape[0:2][::-1]  # bchw...
        output.append(shape.tolist())
    return output


def to_onnx_axis(axis):
    if axis == 0:
        return 0
    elif axis == -1:
        return 1
    else:
        return axis + 1


def np2tensor(data, name):
    return onnx.helper.make_tensor(
            name=name,
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype],
            dims=data.shape,
            vals=data.flatten())


def np2constant(data, name):
    tensor = np2tensor(data, f'{name}/value')
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=tensor)
    return node, tensor