import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logisticregression as logre
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

testlabel[testlabel==0]=-1

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
epochs = 100
lrnrate = 0.01


for var in variances:

	logist = logre.logreg(vals, label)
	
	logist.desc(epochs, var, lrnrate, lrscale=0.34)

	mapres, mlres = logist.pred(testvals)
	
	lrmap, lrml = logist.getweights()
	
	print("\n\n________")
	print("VARIANCE:", var)
	
	print("\nMAP Weight vector:", lrmap)
	print("ML Weight vector:", lrml)
	
	print("\nMAP Accuracy:",((np.sum(mapres==testlabel))/testlabel.size))
	print("ML Accuracy:",((np.sum(mlres==testlabel))/testlabel.size))