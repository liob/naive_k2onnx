naive_k2onnx
============
The goal of naive_k2onnx is to provide a simple tool to convert our Keras models to ONNX. It implements only a small subset of available Keras layers. It does not aim to be a general-purpose Keras to ONNX converter but to cater to specific use cases.


Why yet another Keras to ONNX converter?
----------------------------------------
At the time of conception, there were no available Keras to ONNX converters, which sufficiently served my needs. I.e. [keras2onnx] is excellent, however, we use a lot of custom layers, which lead to a high amount of unnecessary transpose operations in the ONNX compute graph (obviously bad for performance). I fully expect (and hope) that this piece of software will be rendered redundant shortly, as the keras2onnx optimizer gets smarter.


Assumptions & Requirements
--------------------------
  - The keras model has to be [channels last][keras image format] (nhw...c)
  - Tensorflow Keras >= 2.0
  - This software does not work with keras.io
  - The output format is channels first


Inner Workings
--------------
The naive approach is to provide a mapping for each layer to native [onnx operators]. This is achieved by traversing the Keras graph, generating the necessary operators to map the input tensor of the layer to the output tensor.


License
-------
This software is licensed under the terms of the unlicense.


[keras2onnx]: https://github.com/onnx/keras-onnx
[keras image format]: https://www.tensorflow.org/api_docs/python/tf/keras/backend/image_data_format
[onnx operators]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
