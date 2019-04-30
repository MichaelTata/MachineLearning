import pandas as pd
import numpy as np
import svm as supv


#Train data
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"])
vals = vals.values
label = vals[:,-1]

label[label==0]=-1

vals = np.delete(vals, -1, axis=1)

weights = np.zeros(vals.shape[1])

#vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)


#Test data
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)
#testvals = np.concatenate((np.ones([testvals.shape[0],1], dtype=np.int),testvals),axis=1)

#weights = np.zeros(vals.shape[1])
testlabel[testlabel==0]=-1


cfrac = [100/873,500/873,700/873]



for val in cfrac:
		weightsp = np.array(weights)

		vecmac = supv.Svm(vals, weightsp, label)

		vecmac.dual(val, 0, 0)


		av, svidx = vecmac.dualpred(testvals)
		pred = np.sign(av)

		print("\n\n C VAL:", val)
		print("\nAccuracy:",((np.sum(pred==testlabel))/testlabel.size))
		print("________")


print("PART 2")
gammas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]

prev = None

for val in cfrac:

	for gam in gammas:
		weightsp = np.array(weights)

		vecmac = supv.Svm(vals, weightsp, label)

		vecmac.dual(val, gam, 1)

		av, svidx = vecmac.dualpred(testvals)
		pred = np.sign(av)
		
		print("Support Vec Count:", svidx.size)
		if val == 500/873:
			s = 0
			if prev is not None:
				s = np.intersect1d(prev, svidx).size
				
			print("SUPVEC SIMILARITY COUNT:", s)
			prev = svidx
		
		print(" C VAL:", val)
		print("GAMMA VAL:", gam)
		print("Accuracy:",((np.sum(pred==testlabel))/testlabel.size))
		print("________\n")
