import pandas as pd
import numpy as np
import decisiontree as dt
import ensemble as ens
import cProfile, pstats, io
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

#numerical attributes indexes
numerics = np.array([0,5,9,11,12,13,14])
	
attr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
attr =  attr.values
label = attr[:,-1]
fracs = np.ones_like(label)

#convert label to {-1, 1} 
label[label=="no"] = -1
label[label=="yes"] = 1

label = np.array(label, dtype=float)

attr = np.delete(attr, -1, axis=1)

signlbl = np.sign(label)


for i in range(0,numerics.size):
	median = np.median(attr[:,numerics[i]])
		
	for x in range(0,len(attr[:,numerics[i]])):
		
		if attr[x,numerics[i]] < median:
			attr[x,numerics[i]] = 0
		else:	
			attr[x,numerics[i]] = 1
			


#read in and split the training dataset, for prediction testing.
	trainattr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
		
	trainattr = trainattr.values
	trainlabel = trainattr[:,-1]
	trainattr = np.delete(trainattr, -1, axis=1)
	
	
	

testattr = pd.read_csv('./Data/Bank/test.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])		
testattr = testattr.values
testlabel = testattr[:,-1]
testattr = np.delete(testattr, -1, axis=1)
		

trainattr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])		
trainattr = trainattr.values
trainlabel = trainattr[:,-1]
trainattr = np.delete(trainattr, -1, axis=1)
		

for i in range(0,numerics.size):
	median = np.median(testattr[:,numerics[i]])
	median2 = np.median(trainattr[:,numerics[i]])
			
	for x in range(0,len(testattr[:,numerics[i]])):
		if testattr[x,numerics[i]] < median:
			
			testattr[x,numerics[i]] = 0
		else:
			testattr[x,numerics[i]] = 1
	for x in range(0,len(trainattr[:,numerics[i]])):
		if trainattr[x,numerics[i]] < median2:
			trainattr[x,numerics[i]] = 0
		else:
			trainattr[x,numerics[i]] = 1
			
			
testlabel[testlabel=="no"] = -1
testlabel[testlabel=="yes"] = 1
testlabel = np.array(testlabel, dtype=float)			

trainlabel[trainlabel=="no"] = -1
trainlabel[trainlabel=="yes"] = 1
trainlabel = np.array(trainlabel, dtype=float)	


count = attr.shape[0]


#itlist = [1,2,4,8,12,16,20,25,30,40,50,60,90,125,250,500,700,1000]
itlist = [1000]

trainerrs = np.zeros(len(itlist))
testerrs = np.zeros(len(itlist))

idx = 0


for curriter in itlist:
	pererrtest = np.zeros(curriter)
	pererrtrain = np.zeros(curriter)

	tlp = np.array(testlabel)
	trlp = np.array(trainlabel)

			#pr = cProfile.Profile()
			#pr.enable()

	adboost = ens.Adaboost(label, attr, stsize=1,iterations=curriter)
	
	testpred, tsp = adboost.predict(testattr)
	trainpred, trnp = adboost.predict(trainattr)
	
	totaltest = (testpred == np.sign(testlabel)).sum()
	totaltrain = (trainpred == np.sign(trainlabel)).sum()
	
	unqtest,testcnt  = np.unique(testpred, return_counts=True)
	unqtrn,trncnt  = np.unique(trainpred, return_counts=True)
	
	testtr, truetestcnt = np.unique(testlabel, return_counts=True)
	trnttr, truetraincnt = np.unique(trainlabel, return_counts=True)
	
	print("\n__________ITERS:", curriter,"______________")
	
	print("Correct test preds:", totaltest)
	print("Correct train preds:", totaltrain)
	
	print("\nTrain")
	for i in range(len(unqtrn)):
		print("Level",i ,"  Pred:", unqtrn[i], "= ", trncnt[i],"   True:", trnttr[i], "= ", truetraincnt[i])
	print("Test")
	for i in range(len(unqtest)):
		print("Level",i ,"  Pred:", unqtest[i], "= ", testcnt[i],"   True:", testtr[i], "= ", truetestcnt[i])
		
	trainerrs[idx] = 1 - (totaltrain/count)
	testerrs[idx] = 1 - (totaltest/count)
	
	
	if curriter == 1000:
		for i in range(curriter):
			temp = (trnp[i]==np.sign(trainlabel)).sum()
			temp2 = (tsp[i]==np.sign(testlabel)).sum()
			#print(temp)
			#print(temp2)
			pererrtest[i] = 1 - (temp/count)
			pererrtrain[i] = 1 -(temp2/count)
	
	
	
	
	idx = idx + 1
			#pr.disable()
			#s = io.StringIO()
			#sortby = 'cumulative'
			#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
			#ps.print_stats()
			#print(s.getvalue())

print(pererrtest)
print(pererrtrain)
			
plt.plot(np.array(itlist),trainerrs,  color="blue", linewidth=1.0, linestyle="-", label="train error")
plt.plot(np.array(itlist),testerrs,  color="red", linewidth=1.0, linestyle="-", label="test error")
plt.xlabel('Iterations')
plt.ylabel('Errors')
plt.legend(loc='upper left', frameon=False)
plt.show()



plt.plot( range(0,itlist[-1]),pererrtrain, color="blue", linewidth=1.0, linestyle="-", label="train err ")
plt.plot( range(0,itlist[-1]),pererrtest,color="red", linewidth=1.0, linestyle="-", label="test error")

plt.xlabel('Iterations')
plt.ylabel('Errors')
plt.legend(loc='upper left', frameon=False)
plt.show()
			