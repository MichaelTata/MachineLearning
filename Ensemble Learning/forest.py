import pandas as pd
import numpy as np
import decisiontree as dt

attr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
attr =  attr.values
label = attr[:,-1]
attr = np.delete(attr, -1, axis=1)
fracs = np.ones_like(label)


#convert labels to 0 and 1 so we can combine votes for prediction
label[label=="no"] = 0
label[label=="yes"] = 1

label = np.array(label, dtype=float)

numerics = np.array([0,5,9,11,12,13,14])

for i in range(0,numerics.size):
	median = np.median(attr[:,numerics[i]])
		
	for x in range(0,len(attr[:,numerics[i]])):
		
		if attr[x,numerics[i]] < median:
			attr[x,numerics[i]] = 0
		else:	
			attr[x,numerics[i]] = 1
			

#read in and split the testing dataset
testattr =  pd.read_csv('./Data/Bank/test.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
#testattr = pd.read_csv('./Data/temp2train.csv',names=["Outlook","Temperature","Humidity","Wind","label"])
		
testattr = testattr.values
testlabel = testattr[:,-1]
testattr = np.delete(testattr,-1,axis=1)

#read in and split the training dataset, for prediction testing.
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
			
testlabel[testlabel=="no"] = 0
testlabel[testlabel=="yes"] = 1
testlabel = np.array(testlabel, dtype=float)			

trainlabel[trainlabel=="no"] = 0
trainlabel[trainlabel=="yes"] = 1
trainlabel = np.array(trainlabel, dtype=float)	

count = attr.shape[0]
print(count)


#part C
itlist = range(0,10)

singlepreds = []
bagpreds = []

for curriter in itlist:
	origlb = np.array(label)
	origatr = np.array(attr)
	origfra = np.array(fracs)
	
	res = np.zeros(count, dtype=float)
	
	#for i in range(curriter):
	for i in range(1,100):
			
		#, labels, attr, fracs, currlvl, maxd=17, pchoice=None, origs=None, random=None, purityfnc=entropy, parent=None):
		#Draw Our samples
		indexes = np.random.choice(count,1000,replace=False) 
		#print(indexes)
		
		atrpas = attr[indexes,:]
		lblpas = label[indexes]
		fracps = fracs[indexes]
		
		btree = dt.RandTree(lblpas, atrpas, fracps, 1,purityfnc=dt.entropy)
		
		if i == 1:
			singlepreds.append(btree)
			
		tmpr = btree.predict(trainattr)
			
		res = res + tmpr
		
		
	res = res / 100
	
	restree = dt.Tree(origlb,origatr,res,1)
	
	bagpreds.append(restree)
	

sres = np.zeros(count,dtype=float)
tempattr = np.array(attr)
testlb = np.array(label)
testatr = np.array(testattr)

for stree in singlepreds:
	temp = stree.predict(testattr)
	sres = sres + tmpr

print(sres)	


srtree = dt.Tree(origlb,testatr,sres,1)	
avg = sres / len(singlepreds)


svariance = (1/(len(singlepreds)-1))*(np.square(sres-avg))
sbias = np.square(sres - testlabel)

svf = np.sum(svariance)/count
print("Singles Variance:", svf)
sbf = np.sum(sbias)/count
print("Singles Bias:", sbf)
generrsingle = svf + sbf
print("Singles Error:", generrsingle)


print("_________________________")
bres = np.zeros(count,dtype=float)
tempattr = np.array(attr)
testlb = np.array(label)
testatr = np.array(testattr)

for btree in bagpreds:
	temp = btree.predict(testattr)
	bres = bres + tmpr

print(bres)	


brtree = dt.Tree(origlb,testatr,bres,1)	
avg = bres / len(bagpreds)


bvariance = (1/(len(bagpreds)-1))*(np.square(bres-avg))
bbias = np.square(bres - testlabel)

bvf = np.sum(bvariance)/count
print("Bagged Variance:", svf)
bbf = np.sum(bbias)/count
print("Bagged Bias:", bbf)
generrbag = bvf + bbf
print("Bagged Error:", generrbag)
