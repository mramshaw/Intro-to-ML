# Custom Classifier - Machine Learning Recipes #5

Writing and evaluating a custom classifier.

This will be a simple __nearest neighbour__ variant using `scipy.spatial.distance`.

## Starting Point

Our starting point is the classifier from [ML #4](../Pipeline/):

    $ python pipeline.py 
    Prediction accuracy: 0.96
    $

So we are looking for results somewhere in the neighbourhood of 96%.

## Random Classifier

With a random classifier, the prediction accuracy is approximately a third:

    $ python custom_classifier.py 
    Prediction accuracy: 0.306666666667
    $

[There are three flower types, so a random guess should be right about this percentage of the time.]

## Execution

To run, type the following:

    python custom_classifier.py

The results should look something like:

    $ python custom_classifier.py 
    Prediction accuracy: 0.96
    $ python custom_classifier.py 
    Prediction accuracy: 0.946666666667
    $ python custom_classifier.py 
    Prediction accuracy: 0.973333333333
    $ python custom_classifier.py 
    Prediction accuracy: 0.96
    $

So, a pretty good result from a simple (K=1) implementation of K Nearest Neighbour.

## Credits

    https://www.youtube.com/watch?v=AoeEHqVSNOw
