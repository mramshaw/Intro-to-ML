# Handwriting Classifier - Machine Learning Recipes #7

There are a wide variety of installation options for [TensorFlow](https://www.tensorflow.org/install/).

This video specifically deals with the [Docker](https://hub.docker.com/r/tensorflow/tensorflow/) option.

Note that, for GPU-based TensorFlow, there is also an [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) option.

## MNIST Data

Run the following Python code to download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/):

    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)

    learn = tf.contrib.learn

    mnist = learn.datasets.load_dataset('mnist')

The results should look as follows:

    Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
    Extracting MNIST-data/train-images-idx3-ubyte.gz
    Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
    Extracting MNIST-data/train-labels-idx1-ubyte.gz
    Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
    Extracting MNIST-data/t10k-images-idx3-ubyte.gz
    Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
    Extracting MNIST-data/t10k-labels-idx1-ubyte.gz
    $

Now it will be available for subsequent runs.

## Credits

    https://www.youtube.com/watch?v=Gj0iyo265bc
