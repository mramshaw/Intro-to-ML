from numpy import random

N = 500     # number of windows to classify
d = 300     # dimensionality of each window
C = 5       # number of classes

W = random.rand(C, d)

wordvectors_list = [random.rand(d, 1) for i in range(N)]
wordvectors_one_matrix = random.rand(d, N)

# the percent ('%') prefix is only needed if Automagic is NOT on 
#     %timeit [W.dot(wordvectors_list[i]) for i in range(N)]
#     %timeit W.dot(wordvectors_one_matrix)
timeit [W.dot(wordvectors_list[i]) for i in range(N)]
timeit W.dot(wordvectors_one_matrix)
