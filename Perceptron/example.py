import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import perceptron as pr
from matplotlib.pyplot import cm


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



weightsp = np.array(weights)
ptron = pr.perceptron(vals, weightsp, label, epochs, lrnrate)
stw, stdhist = ptron.standard()
standard = np.sign(np.dot(stw, testvals.T))


weightsp = np.array(weights)
ptron = pr.perceptron(vals, weightsp, label, epochs, lrnrate)
awg, avghist = ptron.averaged()
averaged = np.sign(np.dot(awg, testvals.T))






print("Standard Final Weights:", stw)
print("STANDARD MISSES:", testlabel.size - np.sum(standard==testlabel), "   Accuracy:", ((np.sum(standard==testlabel))/testlabel.size))

print("Averaged Final Weights:", awg)
print("AVG MISSES:", testlabel.size - np.sum(averaged==testlabel), "   Accuracy:", ((np.sum(averaged==testlabel))/testlabel.size))