"""
Group 3 (Cody Hartman, Jordan Gahan, Matt Dennie)
Machine Learning 
5/13/17
k-fold validation using custom non-smo SVM.
Feature extraction done using transfer learning with inception v3 CNN.
"""

import numpy as np
import cvxopt.solvers

## lin kernel ---- simple dot
def kernel_linear(x1, x2):
	return np.dot(x1, x2)
class SVM(object):
	def __init__(self, c, kernel):
		## c value for SVM - soft kernel
		self.c = c
		## kernel - only linear is supported or coded right now
		self.kernel = kernel
		
	def fit(self, X, y):
		#X should be a numpy array 
		samples, features = X.shape
		# calculate Gram matrix X=xx' with kernel
		K = self.gram_matrix(X,samples)
		# Solve the lagrange dual problem --- SMO not programmed 
		# Lagrange being solved the slow QP way
		# Could be replaced by solving the KKT conditions
		# TODO
		a = self.compute(K,samples,y)
		
		## 1e-5 limits support vectors
		## a is lagrange mult
		## support vector indices
		sv_i = a > 1e-5
		## get value of sv_i from range of lagrange a
		ind = np.arange(len(a))[sv_i]
		## support multiplier
		self.a = a[sv_i]
		## support vectors
		self.sv = X[sv_i]
		self.sv_y = y[sv_i]
		
		# calculate bias = y_k - sigma a y_i  K(x_k, x_i)
		# bias equation from class modified from weight eq
		
		
		self.b = 0
		
		for n in range(len(self.a)):
			self.b += self.sv_y[n]
			
			self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv_i])
		self.b /= len(self.a)

		# calculate weight
		self.weights = np.zeros(features)
		## slides day 27 pg 16 W = sigma a_i * y_i * x_i
		for n in range(len(self.a)):
			self.weights += self.a[n] * self.sv_y[n] * self.sv[n]
		########################################################
			
	#X = xx'
	## day 28 slides pg 33
	def gram_matrix(self,X,samples):
		K = np.zeros((samples, samples))
		for i in range(samples):
			for j in range(samples):
				K[i,j] = self.kernel(X[i], X[j])
		return K
				
	def compute(self, K, samples, y):
		#p, Q, G, h, A, b are parameters passed into cvxopt solver to solver the QP
		#qp setup was learned from here http://tullo.ch/articles/svm-py/ 
		p = cvxopt.matrix(np.outer(y,y) * K)
		Q = cvxopt.matrix(np.ones(samples) * -1)
		A = cvxopt.matrix(y, (1,samples))
		b = cvxopt.matrix(0.0)
		tmp1 = np.diag(np.ones(samples) * -1)
		tmp2 = np.identity(samples)
		G = cvxopt.matrix(np.concatenate((tmp1, tmp2),axis = 0))
		tmp1 = np.zeros(samples)
		#Soft-margin
		tmp2 = np.ones(samples) * self.c
		## should I be doing a hstack instead of a vstack?
		h = cvxopt.matrix(np.concatenate((tmp1, tmp2),axis = 0))
		solution = cvxopt.solvers.qp(p, Q, G, h, A, b)
		# a is found
		a = np.ravel(solution['x'])
		return a
		
		
	def predict(self, X):
		if self.weights is not None:
			## day 27 pg 16 slides y = sign sigma a_i * y_i * x^t * z + b = (w^t * z) + b
			## z is X in our case
			## z is thing we want to predict
			## take dot product of X and w and add bias
			return np.dot(X, self.weights) + self.b
		else:
			## calculate weights again
			## weights should be calculated already but if not
			
			
			result = self.b
			## compute prediction
			for a, y_k, x_k in zip(self.a, self.sv_y, self.sv):
				result += a * y_k * self.kernel(x_k, X)
			return result

  
		
	
