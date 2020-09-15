# A perceptron is a neural network with one input and one ouput node

'''
f1 f2 f3   target
0 	1 	0 	1
0	0 	1 	0
1 	0 	0 	0 
1 	1 	0 	1 
1 	1 	1	1 

perceptron architecture:

	input(f1) O --w1--\
	input(f2) O --w2---O (output) 
	input(f3) O --w3--/

'''

import numpy as np 

# data
features = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
target = np.array([[1,0,0,1,1]]).reshape(5,1)

# components of perceptron
np.random.seed(7)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
learningrate = 0.01

# activation function
def sigmoid(x):
	return 1/(1+np.exp(-x))

# derivative of activation function
def sigmoid_der(x):
	return sigmoid(x)*(1-sigmoid(x))


# Perceptron

epochs = 100000

for epoch in range(epochs):
	input = features

# feedfoward
	xw = np.dot(features, weights) + bias
	z = sigmoid(xw)

# backpropagation
	error = z - target
	print(error.sum())

	d_cost_by_d_pred = error
	d_pred_by_dz = sigmoid_der(z)

	z_delta = d_cost_by_d_pred * d_pred_by_dz

	inputs = features.T 
	weights = weights - learningrate * np.dot(inputs, z_delta)

	for num in z_delta:
		bias = bias - learningrate * num

'''
# testing
test = np.array([1,0,1])
prediction = sigmoid(np.dot(test, weights) + bias)
print(prediction)

'''










