import pandas as pd
import numpy as np
import decisiontree as dt


## 0 for categorical attributes, 1 for numerical(or attributes based on ranges)
num = 1

##Categorical
if num == 0:
	attr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
	#attr = pd.read_csv('./Data/temp2.csv', names=["Outlook","Temperature","Humidity","Wind","label"])


	attr = attr.values
	label = attr[:,-1]
	attr = np.delete(attr, -1, axis=1)

	#quick test for the purity & gain functions
	#print("Entropy, Gini, MajorityError")
	#for i in range(0,4):
		#print(dt.gain(attr[:,i],label,dt.entropy),"...", dt.gain(attr[:,i],label,dt.gini), "....", dt.gain(attr[:,i],label,dt.majority))

	#(self, labels, attr, currlvl, maxd, origs=None, purityfnc=entropy, parent=None)

	decEnt = dt.Tree(label, attr, 1, 8, None, None, dt.entropy)
	decGi = dt.Tree(label, attr, 1, 8, None, None, dt.gini)
	decErr = dt.Tree(label, attr, 1, 8, None, None, dt.majority)


	testattr = pd.read_csv('./Data/test.csv',names=["buying","maint","doors","persons","lug_boot","safety","label"])
	#testattr = pd.read_csv('./Data/temp2train.csv',names=["Outlook","Temperature","Humidity","Wind","label"])


	testattr = testattr.values
	testlabel = testattr[:,-1]
	testattr = np.delete(testattr,-1,axis=1)



	predictions1 = decEnt.predict(testattr)
	predictions2 = decGi.predict(testattr)
	predictions3 = decErr.predict(testattr)


	total = np.sum(predictions1 == testlabel)
	print("\n\nCorrect Count:", total)
	print("Accuracy Percentage Entropy: ", total/testlabel.shape[0])

	total = np.sum(predictions2 == testlabel)
	print("\nCorrect Count:", total)
	print("Accuracy Percentage Gini Index: ", total/testlabel.shape[0])

	total = np.sum(predictions3 == testlabel)
	print("\nCorrect Count:", total)
	print("Accuracy Percentage Majority Error: ", total/testlabel.shape[0])

##Numerical
else:
	
	#numerical attributes
	numerics = np.array([0,5,9,11,12,13,14])
	
	attr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
	attr =  attr.values
	label = attr[:,-1]
	attr = np.delete(attr, -1, axis=1)
	
	
	#Convert numerics to binary categories based off of median 
	for i in range(0,numerics.size):
		median = np.median(attr[:,numerics[i]])
		#print("MEDIAN:",median)
		#print(attr[:,i])
		for x in range(0,len(attr[:,numerics[i]])):
			if attr[x,numerics[i]] < median:
		
				attr[x,numerics[i]] = 0
			else:
				attr[x,numerics[i]] = 1
				
		#print(attr[:,numerics[i]])
		
	
	decEnt = dt.Tree(label, attr, 1, 8, None, None, dt.entropy)
	decGi = dt.Tree(label, attr, 1, 8, None, None, dt.gini)
	decErr = dt.Tree(label, attr, 1, 8, None, None, dt.majority)

	
	
	
	
	
	testattr = pd.read_csv('./Data/Bank/test.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
	
	testattr = testattr.values
	testlabel = testattr[:,-1]
	testattr = np.delete(testattr, -1, axis=1)
	
	for i in range(0,numerics.size):
		median = np.median(testattr[:,numerics[i]])
		#print("MEDIAN:",median)
		#print(attr[:,i])
		for x in range(0,len(testattr[:,numerics[i]])):
			if testattr[x,numerics[i]] < median:
		
				testattr[x,numerics[i]] = 0
			else:
				testattr[x,numerics[i]] = 1
				

	
	
	predictions1 = decEnt.predict(testattr)
	predictions2 = decGi.predict(testattr)
	predictions3 = decErr.predict(testattr)


	total = np.sum(predictions1 == testlabel)
	print("\n\nCorrect Count:", total)
	print("Accuracy Percentage Entropy: ", total/testlabel.shape[0])

	total = np.sum(predictions2 == testlabel)
	print("\nCorrect Count:", total)
	print("Accuracy Percentage Gini Index: ", total/testlabel.shape[0])

	total = np.sum(predictions3 == testlabel)
	print("\nCorrect Count:", total)
	print("Accuracy Percentage Majority Error: ", total/testlabel.shape[0])

	
	