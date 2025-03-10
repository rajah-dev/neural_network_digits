# Introduction
The following code and content is adapted from [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) and [3Blue1Brown](https://www.3blue1brown.com/).
This was used as an educational activity to better understand and have an under-the-hood look at simple neural networks. 
As such the comments and code itself contain extraneous information, personal notes, and possibly incorrect information

There are some changes due to Python 2 -> Python 3 versioning, but may be bad practice or not optimized. 
I later found that a [forked version](https://github.com/unexploredtest/neural-networks-and-deep-learning) of this already exists. 

## Usage
In the Python 3 Shell
```
> import mnist_loader
> training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```
1. load the `data/mnist.pkl.gz` into training_data, validation_data, test_data

```
> import network
> net = network.Network([784, 16, 16, 10])
> net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
```
1. create a neural network of input layer for 784 (28 x 28 image size) pixels, two hidden layers of 16 nodes, and an output layer of 10 corresponding to digit (0 - 9)
2. use stochastic gradient descent to learn from the training data
    - over 30 epochs, test_data is provided so after each epoch the network will be tested
    - mini-batch size of 10
    - a learning rate of 3.0

## License
MIT License

Copyright (c) 2012-2022 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.