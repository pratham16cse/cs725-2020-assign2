import numpy as np

from nn import *


NUM_FEATURES = 90
NUM_POINTS = 1000

train_size = int(0.8*NUM_POINTS)
dev_size = int(0.1*NUM_POINTS)
test_size = NUM_POINTS - train_size - dev_size

def generate_linear_1():
	'''
	Generates
		X : array of shape [NUM_POINTS, NUM_FEATURES]
		W : array of shape [NUM_FEATURES, 1]
		b : array of shape [1, 1]
		y : array of shape [NUM_POINTS, 1]

	Equation
		y = X[:, 0]
	'''
	X = np.concatenate(
		(
			np.random.uniform(0., 10., size=(NUM_POINTS, 1)),
			np.zeros((NUM_POINTS, NUM_FEATURES-1))
		),
		axis=1
	)
	W = np.ones((NUM_FEATURES, 1))
	b = np.zeros((1, 1))
	y = np.dot(X, W) + b

	return X, y

def generate_linear_2():
	'''
	Generates
		X : array of shape [NUM_POINTS, NUM_FEATURES]
		W : array of shape [NUM_FEATURES, 1]
		b : array of shape [1, 1]
		y : array of shape [NUM_POINTS, 1]

	Equation
		y = 2 * X[:, 0] + 3 * X[:, 1]
	'''
	X = np.concatenate(
		(
			np.random.uniform(0., 10, size=(NUM_POINTS, 2)),
			np.zeros((NUM_POINTS, NUM_FEATURES-2))
		),
		axis=1
	)
	W = np.concatenate((np.array([[2.], [3.]]), np.ones((NUM_FEATURES-2, 1))), axis=0)
	b = np.zeros((1, 1))
	y = np.dot(X, W) + b

	return X, y

data_generators = [
	generate_linear_1,
	generate_linear_2
]

for (i, generator) in enumerate(data_generators):
	X, y = generator()
	
	train_input = X[:train_size]
	train_target = y[:train_size]
	dev_input = X[train_size:train_size+dev_size]
	dev_target = y[train_size:train_size+dev_size]
	test_input = X[train_size+dev_size:]
	test_target = y[train_size+dev_size:]
	
	max_epochs = 500
	batch_size = 32
	
	learning_rate = 0.001
	num_layers =  1
	num_units = 1 if i==0 else 2
	lamda = 0.0 # Regularization Parameter
	
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	#get_test_data_predictions(net, test_input)
