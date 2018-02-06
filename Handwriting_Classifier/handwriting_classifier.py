#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

learn = tf.contrib.learn
# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
