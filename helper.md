# Step-by-step Instructions

## Initialization
First, create weights and biases in the __init__(..) function of the code. This can be done several ways. We provide two options here:
```python
biases = []
weights = []
for i in range(num_layers):

	if i==0:
		# Input layer
		weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, num_units)))
	else:
		# Hidden layer
		weights.append(np.random.uniform(-1, 1, size=(num_units, num_units)))

	biases.append(np.random.uniform(-1, 1, size=(num_units, 1)))

# Output layer
biases.append(np.random.uniform(-1, 1, size=(1, 1)))
weights.append(np.random.uniform(-1, 1, size=(num_units, 1)))
```
We have provided one example in the code itself. You can access the weights and biases using list indices.
For example, weights of input layer are in the first position of the list. Note that python lists indices are zero-based, i.e. first position is at index 0.
```python
input_layer_weights = weights[0]
```
Similarly, weights of first hidden layer are at index 1.
```python
first_hidden_layer_weights = weights[1]
```
In case you need to access weights in reverse order, python lists provide negative indexing. For example, weights of output layer can be accessed using index -1 and weights of last hidden layer can be accessed using index -2.
```python
output_layer_weights = weights[-1]
last_hidden_layer_weights = weights[-2]
```
Few more useful indexing tricks are:
```python
weights[1:] # weights of all layers except input layer
weights[:-1] # weights of all layers except output layer
weights[1:-1] # weights of all layers except input and output layer i.e. weights of all HIDDEN layers
```

Another way to implement weights is:
```python
# Assuming 2 hidden layers
weights = OrderedDict(
  0: np.random.uniform(-1, 1, size=(NUM_FEATS, num_units))
  1: np.random.uniform(-1, 1, size=(num_units, num_units))
  2: np.random.uniform(-1, 1, size=(num_units, num_units))
  3: weights.append(np.random.uniform(-1, 1, size=(num_units, 1)))
)
```
Note that OrderedDict is a dictionary structure which remembers the order in which entries were added to the dictionary. More about OrderedDict [here](https://www.geeksforgeeks.org/ordereddict-in-python/)


## Forward Pass
Once the weights and biases are initialized, you can easily forward propogate throught the network. Do not bother about the loss or error at this stage. Pseudocode for forward pass can be written as:
```python
# Given: Input to network, X
h = X
for w, b in weights, biases:
  h = w*h + b
  h = activation(h)

output = h
return output
```
Please note that this is just a pseudocode, and copy-pasting it won't work. Please undertand the logic here.

## Matrix Dimensions
Let us consider an example of data with `NUM_FEATS=3` and `batch_size=5`, 2-hidden layers and `num_units=4`.
```python
NUM_FEATS = 3
batch_size = 5

num_layers = 2
num_units = 4

X = np.random.normal(size=(batch_size, NUM_FEATS))
target = np.random.normal(size=(batch_size, 1))

```
If we consider a 2-hidden layer network, we can initialize weights and biases as given [here](#Initialization).

Now we can write the forward pass as:
```python
a = X
for i, (w, b) in enumerate(zip(weights, biases)):

  h = np.dot(a, w) + b.T

  if i < len(weights)-1:
    a = relu(h)
  else: # No activation for the output layer
    a = h

pred = a
```
Let us first understand how `zip(..)` and `enumerate(..)` work.
```python

# First Create two lists
l_1 = [10,20,30,40]
l_2 = ['a','b','c','d']

# zip(..) function on two lists works as follows:
for i,j in zip(l_1, l_2):
  print(i,j)
```
Output:
```
10 a
20 b
30 c
40 d
```
So, `zip(l_1, l_2)` returns a list of tuples in which `i`-th tuple is `(l_1[i], l_2[i])`.

Now let's see how `enumerate(..)` works:
```python
# enumerate is used as follows:
for i,j in enumerate(l_2):
  print(i,j)
```
Output:
```
0 a
1 b
2 c
3 d
```
`enumerate(l_2)` returns a list of tuples in which `i`-th tuple is `(i, l_2[i])`.


Next we will parse the line `h = np.dot(h, w) + b.T`. 

Note the `transpose` operation on bias `b`. For `i=0`, shapes of `h` and `w` are `[5,3]` and `[3,4]` respectively.
`np.dot(h, w)` computes the matrix multiplication of `h` and `w`. Hence shape of `np.dot(h, w)` is `[5,4].`

Next we add bias to `np.dot(h, w)`. Note that the shape of bias vector at `i=0` is `[4,1]`. In order to add bias vector to all examples in the batch, we use something called [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
Shape of transpose of `b` is `[1,4]`. The addition of two matrices with shapes `[5,4]` and `[1,4]` returns a matrix with shape `[5,4]`. The addition is performed by copying the bias vector 5 times. Hence we use the transpose here. Transpose of vector `b` is denoted as `b.T`.

## Gradient of Loss w.r.t. Output Layer Weights

Continuing above example, let us try to compute the gradient of the loss function with respect to weights of the output layer.
For that, let's first write a loss function.
We will use mean squared error (mse) loss function.
```python
loss = (target - pred)**2 # Sum of squared errors.
```
Note that `loss` is a numpy array of shape `[5,1]`. Let us expand the `pred` variable. For that, let's track few more variables from the forward pass as follows:
```python
a = X
h_states = []
a_states = []
for i, (w, b) in enumerate(zip(weights, biases)):

  if i==0:
    h_states.append(a) # For input layer, both h and a are same
  else:
    h_states.append(h)
  a_states.append(a)

  h = np.dot(a, w) + b.T

  if i < len(weights)-1:
    a = relu(h)
  else: # No activation for the output layer
    a = h

pred = a

```
In our example, weights of the output layer are denoted as `weights[-1]`.
Now, From the forward pass, we know that `pred` is expressed as
```python
pred = np.dot(a_states[-1], weights[-1]) + biases[-1].T # Convince yourself that this is true.
```
Hence we can express `loss` as
```python
loss = (target - (np.dot(a_states[-1], weights[-1]) + biases[-1].T) )**2 # Sum of squared errors.
```
Now we want to compute gradient of `loss` with respect to weights of the output layer i.e. weights[-1]. Gradient of `loss` w.r.t. weights of the output layer
can be written as
```
loss_gradient = a_states[-1] * (pred - target) # batch_size x num_units
update_gradient = 1./batch_size * np.sum(loss_gradient, axis=0) # num_units
```
Please refer to [this file](gradients.pdf) for a detailed explanation of above expression.
`update_gradient` can be used to update the `weights[-1]` as `weights[-1] = weights - learning_rate*update_gradient`.

# General Guidelines
1. Make sure your code is vectorized, otherwise it will be very slow. Like we have computed gradients w.r.t. weights of output layer for all examples in a
single expression, design gradients of all weight vectors in the same way.
2. Do not get hung up over multiplications and divisions by constants such as 2 or m. These constants do not affect the final performance of the network.
3. For dev, there is no need to have batches, entire dev data can be passed as a single batch.
4. Refer to [this link](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf) to understand more about gradient computation.
