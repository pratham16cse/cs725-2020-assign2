import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.
		'''
		raise NotImplementedError

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Either an input vector or a matrix with input vectors
				placed on columns.
		Returns
		----------
			y : Output of the network.
		'''
		raise NotImplementedError

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Feature vectors of training examples 
				(arranged columnwise in 'X' matrix)
			y : Targets of each example in X
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''
		raise NotImplementedError


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Stochastic Gradient Descent (SGD) based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		raise NotImplementedError

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		raise NotImplementedError


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets
		y_hat : predictions

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	raise NotImplementedError

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets
		y_hat : predictions
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	raise NotImplementedError

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets
		y_hat : predictions

	Returns
	----------
		RMSE between y and y_hat.
	'''
	raise NotImplementedError


def train(
	net, optimizer, lamda,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.
	'''
	raise NotImplementedError

def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.

	Returns
	----------
		predictions: Predictions obtained from the forward pass
	'''
	raise NotImplementedError

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	raise NotImplementedError

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 256


	learning_rate = 0.001
	num_layers = 1
	num_units = 64
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()