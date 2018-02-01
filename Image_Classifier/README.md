# Image Classifier (Deep Learning) - Machine Learning Recipes #6

Evaluating images with [TensorFlow](https://www.tensorflow.org/).

[The recommendation is to have ~100 images per image category.]

For more details, refer to the [TensorFlow for Poets codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0).

__tl;dr__ The __TensorFlow__ (tf) components are pluggable and can be swapped-in, like __Scikit-Learn__ (sklearn) components.

## TensorFlow

#### Installation

Follow this [link](https://www.tensorflow.org/install/) for instructions on installing TensorFlow.

If you are using `pip` and already have TensorFlow installed, but wish to upgrade to the latest version:

    $ pip install --upgrade --user tensorflow

You should probably upgrade TensorBoard as well:

    $ pip install --upgrade --user tensorflow-tensorboard

#### Familiarization

We will need `retrain.py`. If you have not cloned the Git repo, it is available at the following
[link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py).

As recommended, start off by running `retrain` to read (and hopefully understand) the help messages:

    $ python retrain.py -h

Depending on the TensorFlow options you have chosen, there may be problems importing `quant_ops`.
The following [link](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
may well be very helpful, and is probably worth scanning in any case. But, __TensorFlow for Poets__
does not actually use the troublesome quantization (it recommends `mobilenet_0.50_224` as opposed to
`mobilenet_0.50_224_quantized`), so all references to `quant_ops` may safely be commented out (lines
117, 783-784, 791-792, 801-808 inclusive) as follows:

	$ diff -uw retrain.py.orig retrain.py
	--- retrain.py.orig	2018-02-01 12:35:59.776185000 -0800
	+++ retrain.py	2018-02-01 13:25:51.315340991 -0800
	@@ -114,7 +114,7 @@
	 from six.moves import urllib
	 import tensorflow as tf
	 
	-from tensorflow.contrib.quantize.python import quant_ops
	+# from tensorflow.contrib.quantize.python import quant_ops
	 from tensorflow.python.framework import graph_util
	 from tensorflow.python.framework import tensor_shape
	 from tensorflow.python.platform import gfile
	@@ -780,16 +780,16 @@
	           [bottleneck_tensor_size, class_count], stddev=0.001)
	       layer_weights = tf.Variable(initial_value, name='final_weights')
	       if quantize_layer:
	-        quantized_layer_weights = quant_ops.MovingAvgQuantize(
	-            layer_weights, is_training=True)
	+#        quantized_layer_weights = quant_ops.MovingAvgQuantize(
	+#            layer_weights, is_training=True)
	         variable_summaries(quantized_layer_weights)
	 
	       variable_summaries(layer_weights)
	     with tf.name_scope('biases'):
	       layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
	       if quantize_layer:
	-        quantized_layer_biases = quant_ops.MovingAvgQuantize(
	-            layer_biases, is_training=True)
	+#        quantized_layer_biases = quant_ops.MovingAvgQuantize(
	+#            layer_biases, is_training=True)
	         variable_summaries(quantized_layer_biases)
	 
	       variable_summaries(layer_biases)
	@@ -798,14 +798,14 @@
	       if quantize_layer:
	         logits = tf.matmul(bottleneck_input,
	                            quantized_layer_weights) + quantized_layer_biases
	-        logits = quant_ops.MovingAvgQuantize(
	-            logits,
	-            init_min=-32.0,
	-            init_max=32.0,
	-            is_training=True,
	-            num_bits=8,
	-            narrow_range=False,
	-            ema_decay=0.5)
	+#        logits = quant_ops.MovingAvgQuantize(
	+#            logits,
	+#            init_min=-32.0,
	+#            init_max=32.0,
	+#            is_training=True,
	+#            num_bits=8,
	+#            narrow_range=False,
	+#            ema_decay=0.5)
	         tf.summary.histogram('pre_activations', logits)
	       else:
	         logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
	$

[My read on things is that quantization involves using 8-bit operations instead of floating-point.]

If you wish to compile `quant_ops` it is available from the following
[link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/python/quant_ops.py).

Once you have fixed (or not) any quantization issues, you should be presented with voluminous help.

## Execution

To run, type the following:

    python [tbd]

## Credits

    https://www.youtube.com/watch?v=cSKfRcEDGUs
