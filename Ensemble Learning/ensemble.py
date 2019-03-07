import pandas as pd
import numpy as np
import decisiontree as dt


class Adaboost:
	hlist=None	#(list)holds classifier for each iteration
	wlist=None	#(list)holds weights for each iteration
	count=None	#(int) holds count(row count) of examples, labels, and weights
	
	
	#labels    - numpy array which holds the resulting label{-1,1} for each example(Make sure it contains only -1 or 1
	#
	#attr      - numpy array which holds the examples and all its attributes
	#
	#fracs 	   - weights array which holds the fractional weight.Will initialize to uniform distribution if nothing is provided
	#			 do not pass 1's for the weights.
	#
	#stsize    - int type which signifies how big of a classifier you want to use. Default is 1 for a stump.
	#
	#iterations- number of adaboost iterations you want to use to train the classifier. 
	#
	def __init__(self,label, attr, weights=None,stsize=1,iterations=20):
		
		self.count = attr.shape[0]
		
		if weights is None:
			weights = np.ones(self.count, dtype=float)
			weights = weights/self.count 
			#print("Weights:",weights)

		self.hlist = []
		self.wlist = []
		
		frc = None
		##
		#labellist = []

		for t in range(iterations):
			lb = np.array(label)
			atr = np.array(attr)
			if frc is None:
				frc = np.array(weights)
				
			stump = dt.Tree(lb,atr,frc,1,stsize,None,None,dt.entropy)
			
				
			predictions = stump.predict(attr)
			
				
			diff = frc.dot(predictions != lb)
	
			sub = 1 - diff
			ds = (np.log(sub) - np.log(diff))
			alpha = (ds)/2 			

			temp = np.array(alpha * predictions)
			temp = np.sign(temp)
			
			##
			#labellist.append(temp)
			
			
	
			

			exps =  np.exp(-alpha * label * predictions)
				
			frc = frc * exps
			frc = frc / np.sum(frc)
			
			self.hlist.append(stump)	
			self.wlist.append(alpha)
			
		####	
		#for i in range(len(labellist)-1):	
			#print("Difference between next:",np.sum(labellist[i]!=labellist[i+1])) 
		
		return
	
	#returns numpyarray that holds the resulting prediction signs(-,+)
	#and also returns a numpy array holding the preds for each iterations stump so you can check error rate
	#testattr-numpyarray, akin to the one used to train{-1,1}
	def predict(self, testattr):
	
		res = np.zeros(self.count)
		prd = np.zeros(self.count, dtype=object)
		idx = 0
		for tr, wg in zip(self.hlist, self.wlist):
			tpr = tr.predict(testattr)	
			res = res + (wg * tpr)
			prd[idx] = np.sign(tpr)
			idx = idx + 1
			
		res = np.sign(res)
		return res, prd