#!/usr/bin/env python

from sklearn import datasets

iris = datasets.load_iris()

# X = features, y = labels
#
# It's a convention that 'X' be upper-case and 'y' lower-case.
#
# The basic idea is that 'y' is a function of 'X':
#
#    y = f(X)
#
X = iris.data
y = iris.target

# Split our dataset into test & training portions

# Note that the next line is deprecated; replaced with the following line.
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn import neighbors
my_classifier = neighbors.KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# Now print out our prediction accuracy

from sklearn.metrics import accuracy_score
print ("Prediction accuracy: {}".format(accuracy_score(y_test, predictions)))
