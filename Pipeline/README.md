# Features - Machine Learning Recipes #4

[![Known Vulnerabilities](http://snyk.io/test/github/mramshaw/Intro-to-ML/badge.svg?style=plastic&targetFile=Pipeline%2Frequirements.txt)](http://snyk.io/test/github/mramshaw/Intro-to-ML?style=plastic&targetFile=Pipeline%2Frequirements.txt)

Evaluating and comparing classifiers with `sklearn` convenience methods.

[Classifiers can be very easily swapped in or out.]

## Execution

To run, type the following:

    python pipeline.py

There can be a certain amount of randomness to `sklearn.metrics.accuracy_score` as follows:

    $ python pipeline.py 
    Prediction accuracy: 0.946666666667
    $ python pipeline.py 
    Prediction accuracy: 0.96
    $ python pipeline.py 
    Prediction accuracy: 0.96
    $ python pipeline.py 
    Prediction accuracy: 0.946666666667
    $

[This randomness is apparently due to the way test/training data is partitioned.]

The above results were for `sklearn.tree.DecisionTreeClassifier`.

Here are some results from `sklearn.neighbors.KNeighborsClassifier`:

    $ python pipeline.py 
    Prediction accuracy: 0.96
    $ python pipeline.py 
    Prediction accuracy: 0.96
    $ python pipeline.py 
    Prediction accuracy: 0.946666666667
    $ python pipeline.py 
    Prediction accuracy: 0.946666666667
    $

[The two classifiers seem to be producing the same accuracy, so no clear winner.]

## Credits

    https://www.youtube.com/watch?v=84gqSbLcBFE
