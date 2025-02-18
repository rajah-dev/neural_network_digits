"""
network.py

The following code is adapted from Neural Networks and Deep Learning 
http://neuralnetworksanddeeplearning.com/

With some changes due to Python 2 -> Python 3

As an educational activity, the comments contain extraneous information and personal notes
The primary example used will be a neural network of input layer = 784 / hidden layers = 16, 16 / output layer = 10
as described in https://www.3blue1brown.com/lessons/gradient-descent

"""
import numpy as np
import random

class Network(object):

    # sizes is a list, ex: [784, 16, 16, 10]
    def __init__(self, sizes):

        # above ex: num_layers = 4
        self.num_layers = len(sizes)
        
        self.sizes = sizes

        # for biases start afer input layer, ex: create 2d numpy arrays (y,1) of 16, 16, and 10 random biases, for use in column vector format
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # for weights, omits output layer and input layer, and pair lists
        # pairs input layer with 2nd layer, 2nd layer with 3rd, 3rd layer with 4th layer, etc creating weight connections between each as (x, y)
        # for each (x, y) create a 2D numpy array with y rows and x columns
        # weights are stored as lists of numpy matrices
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        # return the output of the network if a is input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    # train the neural network using mini-batch stochastic gradient descent
    # the training_data is a list of tuples (x, y) representing the training inputs and the desired outputs
    # if test_data is provided then the network will be evaluated against the test data after each epoch, and partial progress printed out.  
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    # update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
    # the mini_batch is a list of tuples (x, y), and eta is the learning rate
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    

    # return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
    # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # return the number of test inputs for which the neural network outputs the correct result
    # the neural network's output is assumed to be the index of whichever neuron in the final layer has the highest activation
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # return the vector of partial derivatives \partial C_x / \partial a for the output activations
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

# when the input z is a numpy array, Numpy automatically applies the function sigmoid in vectorized form.
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
