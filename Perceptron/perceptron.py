import pandas as pd
import numpy as np


def StandardPerceptron(vals, weights, labels, epoch, lrnrate):

	for it in range(epoch):
		for i in range(vals.shape[0]):
				
			pred = weights.T.dot(vals[i,:])			
			pred = pred * labels[i]
			
			if pred <= 0:
				weights = weights + lrnrate*vals[i,:]*labels[i]
						
	return weights
	
def MarginPerceptron(vals, weights, labels, epoch, lrnrate, tolerance):


	for it in range(epoch):

		for i in range(vals.shape[0]):
				
			pred = weights.T.dot(vals[i,:])
				
			pred = pred * labels[i]
			
			
			margin = pred / np.linalg.norm(weights)
			
			if pred <= 0 or margin < tolerance:
				weights = weights + lrnrate*vals[i,:]*labels[i]
			
				
	return weights
	
def VotingPerceptron(vals, weights, labels, epoch, lrnrate):
	m = 0
	
	count = 1
	votes = []
	allw = []
	
	for it in range(epoch):
		
		
		for i in range(vals.shape[0]):
				
			pred = weights.T.dot(vals[i,:])
				
			pred = pred * labels[i]
			
			if pred <= 0:
				allw.append(weights)
				votes.append(count)
				weights = weights + lrnrate*vals[i,:]*labels[i]					
				m = m+1
				count = 1
				
			else:
				 count = count + 1
				 
				
	return allw, votes
	
def AveragedPerceptron(vals, weights, labels, epoch, lrnrate):
	avg = 0
	
	for it in range(epoch):
		
		for i in range(vals.shape[0]):
				
			pred = weights.T.dot(vals[i,:])
				
			pred = pred * labels[i]
			if pred <= 0:
				weights = weights + lrnrate*vals[i,:]*labels[i]
			avg = avg + weights
				
	return avg

#vals = pd.read_csv("./input.csv",names=["x1","x2","y"])

#Train data
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"])
vals = vals.values
label = vals[:,-1]

label[label==0]=-1

vals = np.delete(vals, -1, axis=1)
vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)


#Test data
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)
testvals = np.concatenate((np.ones([testvals.shape[0],1], dtype=np.int),testvals),axis=1)

weights = np.zeros(vals.shape[1])

testlabel[testlabel==0]=-1


epochs = 10
lrnrate = 0.2
tolerance = 0.05






#mw = MarginPerceptron(vals, weights, label, epochs, lrnrate, tolerance)
#margin = np.sign(np.dot(mw, testvals.T))






#predictions
stw = StandardPerceptron(vals, weights, label, epochs, lrnrate)

awg = AveragedPerceptron(vals,weights,label,epochs,lrnrate)
vpw, votes = VotingPerceptron(vals,weights,label,epochs,lrnrate)


#Sum all weight and count vectors from vote perceptron to get final prediction
sum = 0
votes = np.array(votes)
cwise = np.zeros(len(vpw[0]))
for i in range(len(votes)):
	wtemp = np.array(vpw[i][:])
	cwise = cwise + (wtemp*votes[i])
	sum = sum + votes[i]*(np.sign(wtemp.dot(testvals.T)))
	
	

	
	
votefinal = np.sign(sum) #final vote

standard = np.sign(np.dot(stw, testvals.T)) #final standard


averaged = np.sign(np.dot(awg, testvals.T)) #final avg


print("Weight Vec Count:", len(votes))

for i in range(len(votes)):
	print("LIST:",vpw[i], "    COUNT:", votes[i])
	

print("\nSUM Vote:", cwise , "  ") #Column wise addition for all weight vectors multiplied with its vote count.. Simply for reference to avg
print("AVG weights:",awg, "  \n")


print("STANDARD Avg Err:", np.sum(np.abs(standard-testlabel))/testlabel.size)
print("VOTES Avg Err:", np.sum(np.abs(votefinal-testlabel))/testlabel.size)
print("AVERAGED Avg Err:", np.sum(np.abs(averaged-testlabel))/testlabel.size)


print("\nStandard Final Weights:", stw)
print("\nAveraged Final Weights:", awg)


#print("STANDARD MISSES:", testlabel.size - np.sum(standard==testlabel))

#print("MARGIN MISSES:", testlabel.size - np.sum(margin==testlabel) )

#print("VOTES MISSES:", testlabel.size - np.sum(votefinal==testlabel))

#print("AVG MISSES:", testlabel.size - np.sum(averaged==testlabel) )



