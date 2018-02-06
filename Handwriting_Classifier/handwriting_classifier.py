#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.set_verbosity(tf.logging.ERROR)

learn = tf.contrib.learn

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images       # 55k
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images   # 10k
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Can slice the data as follows:
# max_examples = 10000
# data = data[:max_examples]
# labels = labels[:max_examples]

# Now lets visualize some of the data

def label_display(i):
    img = test_data[i]
    plt.title('Example {} - Label {}'.format(i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()

def predicted_display(i, predicted, labelled, message):
    img = test_data[i]
    plt.title('Prediction {} - Label {} = {}'.format(predicted, labelled, message))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)
    plt.show()

label_display(0)
label_display(1)
label_display(8)

# The number of elements in each datapoint
print 'Features of each image (28x28) = {}'.format(len(data[0]))

feature_columns = learn.infer_real_valued_columns_from_input(data)
classifier = learn.LinearClassifier(feature_columns=feature_columns, n_classes=10)
classifier.fit(data, labels, batch_size=100, steps=1000)

classifier.evaluate(test_data, test_labels)
print 'Accuracy = {}'.format(classifier.evaluate(test_data, test_labels)["accuracy"])

# Now lets visualize some of the predictions

# print 'Predicted {}, Label: {}'.format(classifier.predict(test_data[0]), test_labels[0])
predicted_display(0, classifier.predict(np.array([test_data[0]], dtype=float), as_iterable=False), test_labels[0], 'CORRECT!')

# print 'Predicted {}, Label: {}'.format(classifier.predict(test_data[8]), test_labels[8])
predicted_display(8, classifier.predict(np.array([test_data[8]], dtype=float), as_iterable=False), test_labels[8], 'WRONG!')

weights = classifier.get_variable_value('linear//weight')
f, axes = plt.subplots(2, 5, figsize=(10,4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(()) # ticks be gone
    a.set_yticks(())
plt.show()
