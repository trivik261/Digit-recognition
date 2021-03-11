import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # TODO
    return max(0,x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    x=int((x>0) * 1)
    return x

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """


    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1
        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input =np.dot(self.input_to_hidden_weights,input_values)+self.biases # TODO (3 by 1 matrix)
        vfunc=np.vectorize(rectified_linear_unit)
        hidden_layer_activation =vfunc(hidden_layer_weighted_input)  # TODO (3 by 1 matrix)
        
        output =np.dot(self.hidden_to_output_weights,hidden_layer_activation) # TODO
        vfunc=np.vectorize(output_layer_activation)
        activated_output =vfunc(output) # TODO
        
        
        
        ### Backpropagation ###

        # Compute gradients
        vfunc=np.vectorize(output_layer_activation_derivative)
        output_layer_error =np.dot((activated_output-y),vfunc(output))# TODO
        
        
        
        vfunc=np.vectorize(rectified_linear_unit_derivative)
        
        
        result1=np.dot(output_layer_error,self.hidden_to_output_weights)
        
        hidden_layer_error =np.multiply(np.transpose(result1),vfunc(hidden_layer_weighted_input)) # TODO (3 by 1 matrix)

        

        bias_gradients =hidden_layer_error # TODO
        hidden_to_output_weight_gradients =np.dot(hidden_layer_activation,output_layer_error)# TODO
        input_to_hidden_weight_gradients =np.dot(hidden_layer_error,np.transpose(input_values)) # TODO
        
        # Use gradients to adjust weights and biases using gradient descent
        self.biases =self.biases-self.learning_rate*bias_gradients # TODO
        self.input_to_hidden_weights =self.input_to_hidden_weights-self.learning_rate*input_to_hidden_weight_gradients # TODO
        self.hidden_to_output_weights=self.hidden_to_output_weights-self.learning_rate*np.transpose(hidden_to_output_weight_gradients)
        #print(np.shape(self.hidden_to_output_weights),np.shape(self.hidden_to_output_weights))
    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights,input_values)+self.biases # TODO (3 by 1 matrix)
        vfunc=np.vectorize(rectified_linear_unit)
        hidden_layer_activation =vfunc(hidden_layer_weighted_input)
        output =np.dot(self.hidden_to_output_weights,hidden_layer_activation) # TODO
        vfunc=np.vectorize(output_layer_activation)
        activated_output =vfunc(output) # TODO

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
