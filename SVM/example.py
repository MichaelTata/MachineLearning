import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import svm as supv
from matplotlib.pyplot import cm


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

#weights = np.zeros(vals.shape[1])
testlabel[testlabel==0]=-1


cfrac = [1/873, 10/873,50/873,100/873,300/873,500/873,700/873]


#epochs = 1500
#lrnrate = 0.2
epochs = 100
lrnrate = 0.6



for val in cfrac:

	weightsp = np.array(weights)
	
	vecmac = supv.Svm(vals, weightsp, label)
	
	prim = vecmac.primal(epochs, lrnrate, val,lrch=2)

	spr = np.sign(np.dot(prim, testvals.T))
	print("\n________")
	print("\n\n C VAL:", val)
	print("Weight vector:",prim)
	print("Accuracy:",((np.sum(spr==testlabel))/testlabel.size))
	
