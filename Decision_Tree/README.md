# Decision Tree - Machine Learning Recipes #8

This is a fun exercise which features a decision tree written in pure Python [as a result of which,
no __requirements.txt__ file - as no requirements].

The code is pretty clever, iterating over the features of a toy dataset and picking
the features that are of the most use in classifying the elements of the dataset.

Of course, the elements of the dataset need to be __labelled__ in order for training
to take place.

It is very easy to extend the data set by adding new elements or features, so that
the code is generalizable to a wide variety of classifications.

There is a very useful [Jupyter notebook](https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb),
which is definitely worth a look.

And there is also some heavily-commented [Python code](https://github.com/random-forests/tutorials/blob/master/decision_tree.py),
which is also worth a very close look.

## Concepts Introduced

This video introduces the concepts of 
[Gini Impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)
(which measures how mixed a dataset is WRT a specific feature) and ___Information Gain___
(which measures how much information a particular diagnostic contributes).

I particularly enjoyed the following (very honest) comments on ___Recursion___:

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.

## Test Data

I extended the dataset as follows:

    training_data = [
        ['Red',    3, 'Round', 'Apple'],
        ['Green',  3, 'Round', 'Apple'],
        ['Yellow', 3, 'Round', 'Apple'],
        ['Red',    1, 'Round', 'Grape'],
        ['Red',    1, 'Round', 'Grape'],
        ['Green',  1, 'Oval',  'Grape'],
        ['Yellow', 3, 'Oval',  'Lemon'],
        ['Green',  3, 'Oval',  'Kiwi' ],
    ]

    # Column labels.
    # These are used only to print the tree.
    header = ["color", "diameter", "shape", "label"]

[I added the 'shape' feature to distinguish between apples and lemons,
 also a red apple, a green grape, and a green kiwi.]

## Execution

Run the following command to execute the code:

    $ python decision_tree.py

## Results

The results should look as follows:

    Is diameter >= 3?
    --> True:
      Is shape == Round?
      --> True:
        Predict {'Apple': 3}
      --> False:
        Is color == Yellow?
        --> True:
          Predict {'Lemon': 1}
        --> False:
          Predict {'Kiwi': 1}
    --> False:
      Predict {'Grape': 3}
    Actual: Apple --> Predicted: {'Apple': '100%'}
    Actual: Apple --> Predicted: {'Apple': '100%'}
    Actual: Apple --> Predicted: {'Apple': '100%'}
    Actual: Grape --> Predicted: {'Grape': '100%'}
    Actual: Grape --> Predicted: {'Grape': '100%'}
    Actual: Grape --> Predicted: {'Grape': '100%'}
    Actual: Lemon --> Predicted: {'Lemon': '100%'}
    Actual: Kiwi --> Predicted: {'Kiwi': '100%'}
    $

## Credits

    https://www.youtube.com/watch?v=LDRbO9a6XPU
