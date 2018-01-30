#!/usr/bin/env python

import numpy as np

from sklearn.datasets import load_iris
from sklearn          import tree

iris = load_iris()

test_idx = [0, 1, 50, 51, 100, 101]

# Training data (reserve some data points)
train_target = np.delete(iris.target, test_idx)
train_data   = np.delete(iris.data,   test_idx, axis=0)

# Testing data (the reserved data points)
test_target = iris.target[test_idx]
test_data   = iris.data  [test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print "Labelled:   " + str(test_target)
print "Classifier: " + str(clf.predict(test_data))

# Visualization code
from sklearn.externals.six import StringIO

dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file="IrisTree.dot",
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)

print ("\n'IrisTree.dot' file created, use 'dot -Tsvg IrisTree.dot -O' to convert to SVG\n")

print ("Features: " + str(iris.feature_names))
print ("Lables:   " + str(iris.target_names))

print ("\nTest [3]: ")
print test_data[3], test_target[3]
