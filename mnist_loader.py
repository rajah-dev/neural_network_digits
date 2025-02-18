"""
mnist_loader.py

A library to load the MNIST image data

The following code is taken from Neural Networks and Deep Learning 
http://neuralnetworksanddeeplearning.com/

With some changes due to Python 2 -> Python 3
"""

import pickle
import gzip
import numpy as np

# Return the MNIST data as a tuple containing the training data, the validation data, and the test data
# The training_data is returned as a tuple with two entries
# The first entry contains the actual training images, a numpy ndarray with 50,000 entries
# Each entry is a numpy ndarray with 784 values, representing the pixels in a single MNIST image
# The second entry is a numpy ndarray containing 50,000 entries, the digit values (0...9) for the corresponding images

# The validation_data and test_data are similar, except each contains only 10,000 images

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)


# Return a tuple containing (training_data, validation_data, test_data)
# Based on load_data, but the format is more convenient for use in our implementation of neural networks
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    # training_data is a list containing 50,000 2-tuples (x, y)
    # x is a 784-dimensional numpy.ndarray containing the input image
    # y is a 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct digit for x
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results)) # converted to list, because in python 3 zip returns an iterator

    # validation_data and test_data are lists containing 10,000 2-tuples (x, y).  
    # x is a 784-dimensional numpy.ndarry containing the input image
    # y is the corresponding classification, i.e., the digit values (integers) corresponding to x
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1])) # see above
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1])) # see above

    return (training_data, validation_data, test_data)

# Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere
# This is used to convert a digit(0...9) into a corresponding desired output from the neural network
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
