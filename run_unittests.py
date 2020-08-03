#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
#from naive_k2onnx.unittests.conv import TestConv3D
#from naive_k2onnx.unittests.batchnorm import TestBatchnorm
from naive_k2onnx.unittests.activation import TestPReLU
from naive_k2onnx.unittests.pooling import TestAveragePooling3D
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-v', action='store_true',
                    help='enable verbose logging')
args = parser.parse_args()


if args.v:
    logging.basicConfig(level=logging.DEBUG)

unittest.main(verbosity=3)
