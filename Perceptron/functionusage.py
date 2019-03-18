import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

def StandardPerceptron(vals, weights, labels, epoch, lrnrate):
	
	examplecount = vals.shape[0]
	history = []
	
	for it in range(epoch):
	
		#Create random index array which we use to shuffle our data.
		indexes = np.random.randint(0,examplecount,size=examplecount)

		for i in indexes:
				
			pred = weights.T.dot(vals[i,:])			
			pred = pred * labels[i]
			
			if pred <= 0:
				weights = weights + lrnrate*vals[i,:]*labels[i]
			
		history.append(weights)
		
	return weights, history
	
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
	history = []
	
	for it in range(epoch):
		
		for i in range(vals.shape[0]):
				
			pred = weights.T.dot(vals[i,:])
				
			pred = pred * labels[i]
			if pred <= 0:
				weights = weights + lrnrate*vals[i,:]*labels[i]
			avg = avg + weights
		
		history.append(avg)
		
	return avg, history

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




weightsp = np.array(weights)
#predictions
stw, stdhist = StandardPerceptron(vals, weightsp, label, epochs, lrnrate)


weightsp = np.array(weights)
awg, avghist = AveragedPerceptron(vals,weightsp,label,epochs,lrnrate)

weightsp = np.array(weights)
vpw, votes = VotingPerceptron(vals,weightsp,label,epochs,lrnrate)


#Sum all weight and count vectors from vote perceptron to get final prediction
allvoteacc = []
sum = 0
votes = np.array(votes)
cwise = np.zeros(len(vpw[0]))

wghavg = 0


### For voted perceptron, calculate predictions and averages, and also sum of the final result
for i in range(len(votes)):
	wtemp = np.array(vpw[i][:])
	weighti = wtemp*votes[i]
	cwise = cwise + (weighti)
	sum = sum + votes[i]*(np.sign(wtemp.dot(testvals.T)))
	
	#Get prediction for weight and vote vector i
	restemp =  np.sign(np.dot(weighti,testvals.T))
	rtacc = (np.sum(restemp==testlabel)) 		    #Get the accuracy(total correct predictions) of current weight and count.
	allvoteacc.append(rtacc)
	
	wghavg = wghavg + rtacc	
wghavg = wghavg / len(votes)	 #average accuracy over all weight updates.
###	
	

###Final Weights for all three perceptrons
votefinal = np.sign(sum) #final vote

standard = np.sign(np.dot(stw, testvals.T)) #final standard

averaged = np.sign(np.dot(awg, testvals.T)) #final avg
###



print("Weight Vec Count:", len(votes))
for i in range(len(votes)):
	print("LIST:",vpw[i], "    COUNT:", votes[i])
	


###Find average error for standard and averaged perceptrons
Avgavgerr = 0
Stdavgerr = 0
for i in range(epochs):
	tempavs = np.sign(np.dot(avghist[i],testvals.T))
	tempsts = np.sign(np.dot(stdhist[i],testvals.T))
	
	tapred = ((np.sum(tempavs==testlabel))/testlabel.size)
	tspred = ((np.sum(tempsts==testlabel))/testlabel.size)
	Avgavgerr = Avgavgerr + tapred
	Stdavgerr = Stdavgerr + tspred
Avgavgerr = Avgavgerr / epochs
Stdavgerr = Stdavgerr / epochs
###

print("\nSUM Vote:", cwise , "  ") #Column wise addition for all weight vectors multiplied with its vote count.. Simply for reference to avg



print("\n____________Final Weights_______________")
print("Standard Final Weights:", stw)
print("Averaged Final Weights:", awg)



print("\n____________Averaged Errors_______________")
print("Averaged Perceptron:", Avgavgerr)
print("Standard Perceptron:", Stdavgerr)


print("\n____________Final Accuracies_______________")
print("STANDARD MISSES:", testlabel.size - np.sum(standard==testlabel), "   Accuracy:", ((np.sum(standard==testlabel))/testlabel.size))

#print("MARGIN MISSES:", testlabel.size - np.sum(margin==testlabel), "   Accuracy:", ((np.sum(standard==testlabel))/testlabel.size) )

print("VOTES MISSES:", testlabel.size - np.sum(votefinal==testlabel), "   Accuracy:", ((np.sum(votefinal==testlabel))/testlabel.size))

print("AVG MISSES:", testlabel.size - np.sum(averaged==testlabel), "   Accuracy:", ((np.sum(averaged==testlabel))/testlabel.size))


plt.plot(range(len(votes)), votes,  color="blue", linewidth=1.0, linestyle="-", label="vote count")
plt.plot(range(len(votes)), allvoteacc,  color="red", linewidth=1.0, linestyle="-", label="accurate count")
plt.xlabel('Iteration #')
plt.ylabel('Good Predictions Count & Vote Count')
plt.legend(loc='upper left', frameon=False)
plt.show()

