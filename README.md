# Intro to Machine Learning - Pattern Recognition for Fun & Profit

## Overview

This is a nice free introduction to Machine Learning with Python.

![xkcd](https://imgs.xkcd.com/comics/machine_learning.png)

Here is how the folks at [nVidia](https://blogs.nvidia.com/blog/2016/07/29/whats-difference-artificial-intelligence-machine-learning-deep-learning-ai/) see the relationship between Artifical Intelligence, Machine Learning and Deep Learning:

![AI_versus_ML_versus_Deep_Learning](/images/Deep_Learning_Icons_R5_PNG.jpg.png)

Towards the beginning of my career, I was interested in AI and joined a society founded by [Donald Michie](https://www.theguardian.com/science/2007/jul/10/uk.obituaries1) - who was then at the University of Edinburgh. I wonder how much things have progressed since then?

Machine Learning is hot right now, and of course the cloud providers have noticed.

Here is Google's Cloud offering:

        https://cloud.google.com/products/machine-learning/

For a more sombre view of things, the following article is worth reading:

        https://www.cio.com/article/3223191/artificial-intelligence/a-practical-guide-to-machine-learning-in-business.html

## Prerequisites

1. Python (either Python 2 or Python 3 - or even __both__)

Chris Manning, Stanford, 3 Apr 2017:

> "Essentially, Python has just become the [lingua franca](https://en.wikipedia.org/wiki/Lingua_franca) of nearly all the
> deep learning toolkits, so that seems the thing to use."

        https://youtu.be/OQQ-W_63UgQ?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&t=2102

2. `pip` (if using Python 2) or `pip3` (if using Python 3)

`pip` (or `pip3`) is the Package manager for Python, much as `npm` is the package manager for the Node JavaScript platform.

## scikit-learn

The course uses this library, which it refers to as `sklearn`.

The latest version may be found here:

        http://scikit-learn.org/stable/

To install this library in multi-user mode (not recommended) with `pip` (replace with `pip3` if using Python 3):

        pip install -U scikit-learn

To install this library in single-user mode (recommended) with `pip` (replace with `pip3` if using Python 3):

        pip install --user scikit-learn

## Libraries

It's not really possible to do much of anything in Python without additional libraries.

Essential libraries include:

* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/index.html)

Useful optional libraries include:

* [matplotlib](https://matplotlib.org/)
* [pandas](http://pandas.pydata.org/)

Verify library presence and version with `pip` as with `scikit-learn`:

        pip list --format=legacy | grep numpy

[Replace `numpy` above as necessary.]

Install the library with `pip` (either multi-user or single-user) as with `scikit-learn` above.

## requirements.txt

Of course, it's also possible (as with __npm__ or __composer__) to install all dependencies in one fell swoop (probably a _best practice_).

Simply list the dependencies in a file (for example `requirements` or `requirements.txt`) and install from it:

        pip install --user -r requirements.txt

[Note the `--user` option, which may be omitted for a Global install, also the `-r` option to specify an input file.]

## TODO

- [ ] Finish course

## Credits

Based upon:

        https://www.udacity.com/course/intro-to-machine-learning--ud120

You can find an interview with co-author Katie Malone here:

        http://www.se-radio.net/2017/03/se-radio-episode-286-katie-malone-intro-to-machine-learning/

## Alternatives

This looks like an interesting option too:

        http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning
