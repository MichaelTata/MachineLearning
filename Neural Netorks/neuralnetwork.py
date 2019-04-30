import numpy as np
import pandas as pd
import numpy.linalg as la

def sigmoid(x):
	temp = 1 + np.exp(-x)
	return 1/temp
	
def sigmoidgrad(x):
	s = sigmoid(x)
	return s*(1-s)

	
class Layer:
	def __init__(self, inputsz, outputsz, mean=0, var=2):
	
		self.into = inputsz
		self.outward = outputsz
		
		self.weights = np.random.normal(mean,var,(self.into,self.outward))
		#self.weights = np.zeros((self.into,self.outward))
		self.bias = np.random.normal(0, 1,(self.outward))
		#self.bias = np.ones(self.outward)
		#self.bias = np.zeros(self.outward)
		
		
	def forwardpass(self,vals):
		return sigmoid(np.dot(vals, self.weights)+self.bias)
	
	
	def backprop(self,vals, grad, lrnrate):
		
		derivin = np.dot(grad,self.weights.T)
		derivw = np.dot(vals.T, grad)
		
		derivb = grad.mean(axis=0)/vals.shape[0]

		self.weights = self.weights + lrnrate * derivw
		self.bias = self.bias + lrnrate * derivb

		return derivin, derivw, derivb

	
class NeuralNet:

	def __init__(self, examplesz, featuresz, layercntp=2, neuronsp=8):
		
		self.neurons = neuronsp
		self.layercount = layercntp
		
		
		self.layers = []
		self.layers.append(Layer(featuresz, self.neurons))

		for i in range(self.layercount):

			self.layers.append(Layer(self.neurons,self.neurons))
		
		self.layers.append(Layer(self.neurons,1))
		self.layercount = len(self.layers)

		
		
		
	def fit(self, valsp, labelsp, epochs=55, lrn=0.0001,lrscale=0.34,lsched=None):
	
		
		for it in range(epochs):
		
			curracc = 0
			if lsched is None:
				gamma = lrn/(1+(lrn/lrscale)*it)		
			else:
				gamma = lrn
			indexes = np.random.randint(0,valsp.shape[0]-1,size=valsp.shape[0])
			
			for j in range(0,indexes.size,2):
				
				vals = valsp[indexes[j]:indexes[j]+1,:]
				labels = labelsp[indexes[j]:indexes[j]+1]
				
				layersout = []
				temp = vals
				for i in range(self.layercount):
					layersout.append(self.layers[i].forwardpass(temp))
					temp = layersout[i] 
				
				out = layersout[-1]
				layerinp = [vals] + layersout
				
				err = labels - layersout[-1]
				
				ergrd = err * sigmoidgrad(layersout[-1])
				
				for i in reversed(range(self.layercount)):
					ergrd, dw, db = self.layers[i].backprop(layerinp[i], ergrd, gamma)	

			

		temp = valsp
		for i in range(self.layercount):
			temp = self.layers[i].forwardpass(temp)

		temp2 = np.where(temp >= 0.5, 1.0, -1.0)
		print("Train Accuracy:",((np.sum(temp2==labelsp))/labelsp.size))
		
	def predict(self,vals,labels):
		
		temp = vals
		for i in range(self.layercount):
			temp = self.layers[i].forwardpass(temp)

		
		temp2 = np.where(temp >= 0.5, 1.0, -1.0)
		#print(temp2)
		print("Test Accuracy:",((np.sum(temp2==labels))/labels.size))



#Train data
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"])
vals = vals.values
label = vals[:,-1]

label[label==0]=-1

vals = np.delete(vals, -1, axis=1)
weights = np.zeros(vals.shape[1])
vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)


#Test data
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)
testvals = np.concatenate((np.ones([testvals.shape[0],1], dtype=np.int),testvals),axis=1)


testlabel[testlabel==0]=-1
label = np.reshape(label, (label.size,1))  			   
testlabel = np.reshape(testlabel, (testlabel.size,1))  

nodes = [5, 10, 25, 50, 100]

for i in nodes:
	print("\nNeuron Count/Width:", i)
	#example, feature, layers, neurons
	model = NeuralNet(vals.shape[1],vals.shape[1],1,neuronsp=i)
	model.fit(vals,label)
	model.predict(testvals, testlabel)

