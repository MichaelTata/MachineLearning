import pandas as pd
import numpy as np
import decisiontree as dt


attr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
#attr = pd.read_csv('./Data/temp2.csv', names=["Outlook","Temperature","Humidity","Wind","label"])


attr = attr.values
label = attr[:,-1]
attr = np.delete(attr, -1, axis=1)

#quick test for the purity & gain functions
#print("Entropy, Gini, MajorityError")
for i in range(0,4):
	print(dt.gain(attr[:,i],label,dt.entropy),"...", dt.gain(attr[:,i],label,dt.gini), "....", dt.gain(attr[:,i],label,dt.majority))

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

