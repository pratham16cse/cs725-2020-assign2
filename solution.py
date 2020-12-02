import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

NUM_FEATS = 90

def relu(s):
	"""
	Compute relu activation of s
		relu(s) = max(s,0)
	"""
	return np.where(s>=0, s, np.zeros_like(s))

def relu_prime(s):
	"""
	Compute derivative of relu activation of s
		relu(s) = max(s,0)
		relu_prime(s) = 1. if s>0 else 0
	"""
	return np.where(s>=0, np.ones_like(s), np.zeros_like(s))


class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		a = X
		for layer in range(self.num_layers):
			W, b = self.weights[layer], self.biases[layer]
			h = np.dot(a, W)+b.T
			a = relu(h)

		W, b = self.weights[-1], self.biases[-1]
		pred = np.dot(a, W)+b.T
		return pred

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing bacward pass.
		'''

		h_list, a_list = [], []
		a = X
		h_list.append(X)
		a_list.append(X)
		for layer in range(self.num_layers):
			W, b = self.weights[layer], self.biases[layer]
			h = np.dot(a, W)+b.T
			a = relu(h)
			h_list.append(h)
			a_list.append(a)

		W, b = self.weights[-1], self.biases[-1]
		y_hat = np.dot(a, W)+b.T
		h_list.append(y_hat)
		a_list.append(y_hat)


		da = []
		da.append(2*1/X.shape[0]*(a_list[-1]-y))
		da.append(np.dot(da[-1], self.weights[-1].T))
		for r in range(self.num_layers-1, 0, -1):
			da_r = np.dot(da[-1] * relu_prime(a_list[r+1]), self.weights[r].T)
			da.append(da_r)


		da.reverse()
		dW, db = [], []
		for r in range(self.num_layers):
			dW_r = np.dot(a_list[r].T, da[r] * relu_prime(a_list[r+1])) + lamda * self.weights[r]
			db_r = np.dot(np.ones((1, a_list[r+1].shape[0])), da[r] * relu_prime(a_list[r+1])).T + lamda * self.biases[r]
			dW.append(dW_r)
			db.append(db_r)


		dW.append(np.dot(a_list[-2].T, da[-1]) + lamda * self.weights[-1])
		db.append(np.dot(np.ones((1, a_list[-1].shape[0])), da[-1]).T)

		return dW, db


class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		weights_updated = []
		biases_updated = []
		for W, b, dW, db in zip(weights, biases, delta_weights, delta_biases):
			weights_updated.append(W - self.learning_rate * dW)
			biases_updated.append(b - self.learning_rate * db)

		return weights_updated, biases_updated


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	return np.mean(np.square(y-y_hat))

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
	reg_loss = 0.
	for W, b in zip(weights, biases):
		reg_loss += (np.sum(np.square(W)) + np.sum(np.square(b)))
	return reg_loss

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	return loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	return np.sqrt(np.mean(np.square(y-y_hat)))


def train(
	net, optimizer, lamda, batch_size, max_epochs,
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

	m = train_input.shape[0]

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss

			#print(e, i, rmse(batch_target, pred), batch_loss)

		#print(e, epoch_loss)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` epochs, compute the RMSE on dev data.
	train_pred = net(train_input)
	train_rmse = rmse(train_target, train_pred)
	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)

	print('{}, {}, {}, {}, {:.5f}, {:.5f}'.format(
			optimizer.learning_rate, net.num_layers, net.num_units, lamda, train_rmse, dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	raise NotImplementedError

def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	train = pd.read_csv('data/dataset/train.csv')
	dev = pd.read_csv('data/dataset/dev.csv')
	test = pd.read_csv('data/dataset/test.csv')

	train_input = train.values[:, 1:]
	train_target = train.values[:, 0:1]
	dev_input = dev.values[:, 1:]
	dev_target = dev.values[:, 0:1]
	test_input = test.values

	return train_input, train_target, dev_input, dev_target, test_input


def main():

	# These parameters should be fixed for Part 1
	max_epochs = 50
	batch_size = 128


	learning_rate = float(sys.argv[1])
	num_layers = int(sys.argv[2])
	num_units = int(sys.argv[3])
	lamda = float(sys.argv[4]) # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	#get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
