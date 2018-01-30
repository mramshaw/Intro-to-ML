#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labradors  = 500

# assume both populations are normally-distributed
gh_height  = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(greyhounds)

# Greyhounds red, Labradors blue
plt.hist([gh_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()
