import numpy as np
import numpy.linalg as la
import cvxopt


def gaussiankernel(x1, x2, gamma=3):

	temp = -la.norm(x1 - x2) ** 2
	ker = np.exp(temp/gamma)
	
	return ker


def linkernel(x1, x2, garb):

	ker = np.dot(x1, x2)

	return ker


#######################

class Svm:

##########################################

	def __init__(self, pvals, pweights, plabels):

		self.vals = pvals
		self.weights = pweights
		self.labels = plabels

		self.exsize = plabels.size
		self.ftrsize = pvals.shape[1]

		####Dual Exclusive
		self.alphas = None
		self.alphidx = None
		self.supports = None
		self.bias = 0
		self.choice = 0
		self.gamma = None


	def dualpred(self, xtest):
		#Default Kernel/Linear SVM. Weights
		if self.choice == 0:
			self.weights=np.zeros(self.ftrsize)
			for i in range(self.alphas.size):
				self.weights += self.alphas[i] * self.ysup[i] * self.xsup[i]
			return np.dot(xtest,self.weights) + self.bias, self.alphidx
			
		#Gaussian/Custom Kernel SVM.. No Weight
		else:
			pred = np.zeros(xtest.shape[0])
			for i in range(xtest.shape[0]):
				sum = 0
				for j in range(self.alphas.size):			
					sum += self.alphas[j] * self.ysup[j] * self.kern(xtest[i], self.xsup[j], self.gamma)
				pred[i] = sum
			return (pred + self.bias), self.alphidx
			
	######################################################

	#soft SVM-Dual form, with Kernel options
	def dual(self, cmax, pgamma, kernch, custom=None):

		self.gamma = pgamma

		self.choice = kernch

		if custom is not None:
			self.choice = 2
		
		if kernch == 0:
			self.kern = linkernel

		elif kernch == 1:
			self.kern = gaussiankernel

		else:
			self.kern = custom


		xp = self.vals
		y = self.labels
		

		#compute kernel
		pcx = np.zeros((self.exsize,self.exsize))
		for i in range(self.exsize):
			for j in range(self.exsize):
				pcx[i,j] = self.kern(xp[i], xp[j], self.gamma)
		

		cvxopt.solvers.options['show_progress'] = False
		
		#Set up matrix parameters for optimization solver
		A = cvxopt.matrix(y, (1,self.exsize))
		b = cvxopt.matrix(0.0)
		P = cvxopt.matrix(np.ones((self.exsize,self.exsize)) * pcx)
		q = cvxopt.matrix(np.ones(self.exsize) * -1)
		


		#Set up constraint matrix
		nd = np.diag(np.ones(self.exsize) * -1)
		iden = np.identity(self.exsize)
		
		#constraint matrix
		G = cvxopt.matrix(np.vstack((nd, iden)))
		#lower bound
		nz = np.zeros(self.exsize)
		#upper
		uz = np.ones(self.exsize) * cmax
		h = cvxopt.matrix(np.hstack((nz, uz)))

		#Solve for our alphas
		res = cvxopt.solvers.qp(P, q, G, h, A, b)

		
		opans = np.array(res['x']).flatten()
		
		#Only take the support vector examples(IE:alpha greater than 0)
		validsv = opans > 0.00005
		self.alphidx = np.where(validsv == True)[0]
		self.alphas = opans[validsv]
		self.xsup = self.vals[validsv]
		self.ysup = self.labels[validsv]
		
		
		#calc w(If applicable) and b
		#for each support vector. 
		for i in range(self.alphas.size):
			#Get kernel values for this alpha and example
			temp = pcx[self.alphidx[i], validsv]
			self.bias += self.ysup[i] - np.sum(self.alphas*self.ysup*temp)

		self.bias /= self.alphas.size
	

######################################################	
	
	#soft SVM in primal domain: Using Stochastic Gradient Descent
	def primal(self, epoch, lrnrate, cmax, lrch=1):
		
		weights = np.zeros(self.vals.shape[1])
		examplecount = self.vals.shape[0]
		
		for it in range(epoch):
			
			#P1
			if lrch == 1:
				gamma = lrnrate/(1+(lrnrate/cmax)*it)
			
			#P2
			if lrch != 1:
				gamma = lrnrate/(1+it)
			
			#Create random index array which we use to shuffle our data.
			indexes = np.random.randint(0,examplecount,size=examplecount)

			for i in indexes:
					
				pred = weights.T.dot(self.vals[i,:])			
				pred = pred * self.labels[i]
				
				if pred <= 1: #MAX != 0
				
					nw = np.array(self.weights)
					tempw = np.insert(nw, 0, 0, axis=0)
					weights = (1-gamma)*tempw + (gamma * cmax * examplecount * self.labels[i] * self.vals[i,:])
									
				else: #Reset Weights... MAX = 0
					
					self.weights = (1-gamma)*self.weights
					
		return weights


