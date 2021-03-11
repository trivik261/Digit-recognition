# Digit-recognition
Part-1:

A digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database (as part of the edx course:MITx 6.86x
Machine Learning with Python-From Linear Models to Deep Learning:https://learning.edx.org/course/course-v1:MITx+6.86x+3T2020/home) and using ml concepts that includes linear and logistic regression, non-linear features, regularization, and kernel tricks 

The MNIST database contains binary images of handwritten digits commonly used to train image processing systems. The digits were collected from among Census Bureau employees and high school students. The database contains 60,000 training digits and 10,000 testing digits, all of which have been size-normalized and centered in a fixed-size image of 28 × 28 pixels. Many methods have been tested with this dataset and in this project

Project contains the various data files in the Dataset directory, along with the following python files:

part1/linear_regression.py where you will implement linear regression
part1/svm.py where you will implement support vector machine
part1/softmax.py where you will implement multinomial regression
part1/features.py where you will implement principal component analysis (PCA) dimensionality reduction
part1/kernel.py where you will implement polynomial and Gaussian RBF kernels
part1/main.py where you will use the code you write for this part of the project
Important: The archive also contains files for the second part of the MNIST project. For this project, you will only work with the part1 folder.

To get warmed up to the MNIST data set run python main.py. This file provides code that reads the data from mnist.pkl.gz by calling the function get_MNIST_data that is provided for you in utils.py. The call to get_MNIST_data returns Numpy arrays:

train_x : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length  784=282 . The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).

train_y : The labels for each training datapoint, also known as the digit shown in the corresponding image (a number between 0-9).
test_x : A matrix of the test data, formatted like train_x.

test_y : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.
Next, we call the function plot_images to display the first 20 images of the training set. Look at these images and get a feel for the data (don't include these in your write-up).

Part-2:Implementing a neural network to classify MNIST digits.  

Setup:
Used Python's NumPy numerical library for handling arrays and array operations; used matplotlib for producing figures and plots.

Note on software: For all the projects, we will use python 3.6 augmented with the NumPy numerical toolbox, the matplotlib plotting toolbox. For This project, we will also be using PyTorch for implementing the Neural Nets and scipy to handle sparse matrices.

Download mnist.tar.gz and untar it in to a working directory. The archive contains the various data files in the Dataset directory, along with the following python files:

part2-nn/neural_nets.py  to implement neural net from scratch
part2-mnist/nnet_fc.py  used PyTorch to classify MNIST digits
part2-mnist/nnet_conv.py used convolutional layers to boost performance
part2-twodigit/mlp.py and part2-twodigit/conv.py which are for a new, more difficult version of the MNIST dataset
