#!/usr/bin/env python

from sklearn import tree

# --------------- Training data --------------

# features = [[140, "Smooth"], [130, "Smooth"], [150, "Bumpy"], [170, "Bumpy"]]
#
# 0 = Bumpy, 1 = Smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]

# labels = ["Apple", "Apple", "Orange", "Orange"]
#
# 0 = Apple, 1 = Orange
labels = [0, 0, 1, 1]

# ---------------------------------------------

# "classifier" is usually abbreviated "clf"
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Evaluation data is this single data point
print ("{}, where 0 = Apple and 1 = Orange".format(clf.predict([[160, 0]])))
