#!/usr/bin/env python

import random

from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)

class NearestNeighbour():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
#           label = random.choice(y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index    = i
        return self.y_train[best_index]

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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Custom Classifier

my_classifier = NearestNeighbour()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# Now print out our prediction accuracy

from sklearn.metrics import accuracy_score
print ("Prediction accuracy: {}".format(accuracy_score(y_test, predictions)))
