import numpy as np
from numpy import linalg as LA
from random import randint
import random
import matplotlib.pyplot as plt

	
	
	
#Batch Gradient Descent	
def batchGrad(x, y, weights=None, lrnrate=0.001, tolerance=0.000001, maxit=30000):

	#scale = np.sum(x,axis=0)
	scale = 1
	
	#if weights is None:
	#	weights = np.zeros(x.shape[1])
	
	#if np.all(np.sum(x/scale,axis=0)==1):
	#	print("Already scaled")
	#else:
	#	x = x/scale
	
	#weights = np.array([[0,0,0,0]])
	#weights = np.array([[-1,-1,1,-1]])
	weights = np.array([[1,1/2,-1/2,1/2]])

	it = 0	
	weights = weights*scale
	print(weights)
	weights = weights.T	
	print(weights)
	change = float("inf")
	errc = 0
	lrnrate = 1
	errs = []
	
	while change > tolerance and it < maxit:
		
		prev = weights
		
		wx = x @ weights
				
		weights = weights - lrnrate * (-1*(x.T @ (y-wx)))
		
	
		
		errc = LA.norm(weights - prev)		

		errs.append(errc)
		change = errc
		
		
	
		it = it+1
	
	plt.plot(errs)
	plt.show()
	gcs = np.sum((y - (x @ weights))**2)/2
	print("fin cost", gcs)
	#Extract true weights(needed due to mat and vector mults and broadcast). we need only
	#corresponding elements of diag
	return np.diag(weights/scale), it



#Stochastic Gradient Descent
def stocGrad(x, y, weights=None, lrnrate=0.001, tolerance=0.000001, maxit=50000):	
	
	#scale = np.sum(x,axis=0)
	scale = 1
	
	
	
	if weights is None:
		weights = np.zeros(x.shape[1])
	
	#if np.all(np.sum(x/scale,axis=0)==1):
	#	print("Already scaled")
	#else:
	#	x = x/scale

	#iterator
	it = 0
	
	#randomization values
	itr = 0		
	count = x.shape[0]	
	source = np.random.choice(x.shape[0], x.shape[0], replace=False)
	
	#set up weights
	weights = weights*scale
	weights = weights.T	
	change = float("inf")
	errc = 0
	prev = 0
	
	#errs = []
	
	costs = []
	
	while change > tolerance and it < maxit:
		
		idx = source[itr]
		
		prev = errc
		
		wx = x[idx].dot(weights)	
		
		for pd in range(weights.shape[0]):			
			weights[pd] = weights[pd] - lrnrate * (-1*(x[idx,pd] * ((y[idx]-wx))))		
		
		print("WEIGHTS:",weights)
		
		errc = np.sum((y - (x @ weights))**2)/2
		costs.append(errc)
		
		change = abs(prev - errc)
		
		itr = itr+1
		if itr==count:
			source = np.random.choice(x.shape[0], x.shape[0], replace=False)
			itr = 0
		
		it = it + 1
	
	plt.plot(costs)
	plt.show()
	
	return np.diag(weights/scale), costs, it