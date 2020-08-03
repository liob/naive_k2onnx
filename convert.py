#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import onnx
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from naive_k2onnx import Converter
import argparse


parser = argparse.ArgumentParser(description='convert keras saved model to onnx')
parser.add_argument('-m', type=str, required=True,
                    help='keras model file')
parser.add_argument('-o', type=str, required=False,
                    help='onnx output path')
args = parser.parse_args()


if args.o == None:
    root, cn = path.split(args.m)
    cnx = path.splitext(cn)[0]
    args.o = f'{cnx}.onnx'

keras_model = load_model(args.m)

converter = Converter()
onnx_model = converter.convert(keras_model)

onnx.save(onnx_model, args.o)
