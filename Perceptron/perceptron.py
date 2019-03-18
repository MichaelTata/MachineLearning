import pandas as pd
import numpy as np


class perceptron:
	
	
	
	def __init__(self, pvals, pweights, plabels, pepoch, plrnrate):	
		self.vals = pvals
		self.weights = pweights
		self.labels = plabels
		self.epoch = pepoch
		self.lrnrate = plrnrate
		


	
	def standard(self):
		
		examplecount = self.vals.shape[0]
		history = []
		
		for it in range(self.epoch):
		
			#Create random index array which we use to shuffle our data.
			indexes = np.random.randint(0,examplecount,size=examplecount)

			for i in indexes:
					
				pred = self.weights.T.dot(self.vals[i,:])			
				pred = pred * self.labels[i]
				
				if pred <= 0:
					self.weights = self.weights + self.lrnrate*self.vals[i,:]*self.labels[i]
				
			history.append(self.weights)
			
		return self.weights, history
	

	
	def voting(self):
		m = 0
		
		count = 1
		votes = []
		allw = []
		
		for it in range(self.epoch):
			
			
			for i in range(self.vals.shape[0]):
					
				pred = self.weights.T.dot(self.vals[i,:])
					
				pred = pred * self.labels[i]
				
				if pred <= 0:
					allw.append(self.weights)
					votes.append(count)
					self.weights = self.weights + self.lrnrate*self.vals[i,:]*self.labels[i]					
					m = m+1
					count = 1
					
				else:
					 count = count + 1
					 
					
		return allw, votes
	
	
	def averaged(self):
		avg = 0
		history = []
		
		for it in range(self.epoch):
			
			for i in range(self.vals.shape[0]):
					
				pred = self.weights.T.dot(self.vals[i,:])
					
				pred = pred * self.labels[i]
				if pred <= 0:
					self.weights = self.weights + self.lrnrate*self.vals[i,:]*self.labels[i]
				avg = avg + self.weights
			
			history.append(avg)
			
		return avg, history


#########################################

def margin(self, tolerance):


		for it in range(self.epoch):

			for i in range(self.vals.shape[0]):
					
				pred = self.weights.T.dot(self.vals[i,:])
					
				pred = pred * self.labels[i]
				
				
				margin = pred / np.linalg.norm(self.weights)
				
				if pred <= 0 or margin < tolerance:
					self.weights = self.weights + self.lrnrate*self.vals[i,:]*self.labels[i]
				
					
		return self.weights
		
