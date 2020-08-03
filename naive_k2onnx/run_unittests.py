#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import unittest
from unittests.conv import TestConv3D
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-v', action='store_true',
                    help='enable verbose logging')
args = parser.parse_args()


if args.v:
    logging.basicConfig(level=logging.DEBUG)


unittest.main(verbosity=3)