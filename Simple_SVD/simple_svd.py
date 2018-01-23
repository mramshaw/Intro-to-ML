#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt

la = np.linalg

words = ["I",    "like",     "enjoy",  
         "Deep", "Learning", "NLP",   "flying", "."]

X = np.array([[0,2,1,0,0,0,0,0],
              [2,0,0,1,0,1,0,0],
              [1,0,0,0,0,0,1,0],
              [0,1,0,0,1,0,0,0],
              [0,0,0,1,0,0,0,1],
              [0,1,0,0,0,0,0,1],
              [0,0,1,0,0,0,0,1],
              [0,0,0,0,1,1,1,0]])

U, s, Vh = la.svd(X, full_matrices=False)

plt.axis([-0.8, 0.2, -0.8, 0.9])

for i in xrange(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])

plt.subplots_adjust(left=0.10)
plt.subplots_adjust(right=0.95)
plt.subplots_adjust(top=0.95)
plt.show()
