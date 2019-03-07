import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
import GradDesc as gd

cols=["Cement","Slag","Fly Ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Out"]

#
#dframe = pd.read_csv("./Data/concrete/train.csv",names=["Cement","Slag","Fly Ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Out"])
#dframe = pd.read_csv("./Data/concrete/train.csv",names=["Cement","Slag","Fly Ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "Out"])
dframe = pd.read_csv("./Data/temp.csv",names=["x1","x2","x3","Out"])
data = dframe.values
colcount = data.shape[1]
rowcount = data.shape[0]

output = data[:,-1]

vals = data[:,0:colcount-1]
print(vals)
output = np.reshape(output, (vals.shape[0],1))
vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)

alphas = np.ones([1,colcount])

iterations = 40000
#lrnrates = [0.02688, 0.026903, 0.026907, 0.015]
lrnrates = [0.026903]

#Normal Equation to find optimal(Only should be used if data is small enough)
finans = LA.inv(vals.T @ vals) @ vals.T @ output
print("Optimal Weights", finans.T)

for lrnrate in lrnrates:
	batalpha, itr = gd.batchGrad(vals,output,alphas,lrnrate,0.000001,iterations)
	print("\nBatch Result:", batalpha, " Converged in:", itr , " iterations")
	

print("\n\n")
#lrnrates = [0.1, 0.05, 0.4,0.09]
lrnrates = [0.1]
iterations = 5

for lrnrate in lrnrates:
	stocalpha, costs, itr = gd.stocGrad(vals,output,alphas,lrnrate,0.000001,iterations)
	print("\nStochastic Result:", stocalpha, " Converged in:", itr , " iterations")
	print("fin error:",costs[-1])
	#plt.plot(costs)
	#plt.show()


