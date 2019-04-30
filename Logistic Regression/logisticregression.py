import numpy as np
import numpy.linalg as la

#sigmoid function used for our logistic reg
def sigmoid(x):
	temp = 1 + np.exp(x) 
	return 1/temp

#Logistic Regression using Stochastic Gradient Descent
#Computes both the Maximum A Posterior(MAP) solution
#And the Maximum Likelihood solution(ML)
class logreg:

	def __init__(self, pvals, plabels, pweights=None):

		self.vals = pvals
		self.labels = plabels
		self.exsize = plabels.size
		self.ftrsize = pvals.shape[1]
		
		if pweights is None:
			self.mapweights = np.zeros(self.vals.shape[1])
			self.mlweights = np.zeros(self.vals.shape[1])
			
		else:
			self.mapweights = np.array(pweights)
			self.mlweights = np.array(pweights)
			
	
	#Returns the map weights followed by the ML weights
	def getweights(self):
		return self.mapweights, self.mlweights
	
	#Returns the predictions for the MAP, then the predictions for the ML
	def pred(self, xtest):
	
		mappr = sigmoid(np.dot(xtest,self.mapweights))
		mlpr = sigmoid(np.dot(xtest, self.mlweights))
		
		#determined by sigmoid, if greater than half we have 1, otherwise we have -1.
		#(Could also have taken sign of (weights.T x))
		mapped = np.where(mappr >= 0.5, 1.0, -1.0)
		mled = np.where(mlpr >= 0.5, 1.0, -1.0)
		
		return mapped, mled 
		
	def desc(self, epoch, var, lrnrate=0.05, lsched=None, lrscale=0.34):
	
		for it in range(epoch):	
			#Learning rate schedule. None passed defaults to first below.
			#Otherwise, calls passed function with learning rate, variance, and current iteration as params
			if lsched is None:
				gamma = lrnrate/(1+(lrnrate/lrscale)*it)		
			else:
				gamma = lsched(lrnrate, var, it)
			
			#Create random index array which we use to shuffle our data.
			indexes = np.random.randint(0,self.exsize,size=self.exsize)


			for i in indexes:
				
				#Sigmoid function on our weights,vals,labels
				maptemp = sigmoid(-self.labels[i] * np.dot(self.mapweights, self.vals[i,:]))			
				mltemp = sigmoid(-self.labels[i] * np.dot(self.mlweights, self.vals[i,:]))
				
				#Compute prior for MAP
				prior = self.mapweights/(var)
	
				#Compute Gradients for our MAP and ML
				mapgrd = (self.exsize)*(maptemp*self.vals[i,:]*self.labels[i])/(1+maptemp)
				mlgrd = (self.exsize)*(mltemp*self.vals[i,:]*self.labels[i])/(1+mltemp)
				
				#Update the weights for MAP and ML
				self.mapweights = self.mapweights - gamma  * ((mapgrd - prior)/self.exsize)
				self.mlweights = self.mlweights - ((gamma  * mlgrd)/self.exsize)
					

		