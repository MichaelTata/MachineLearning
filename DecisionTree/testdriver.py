import pandas as pd
import numpy as np
import decisiontree as dt


#attr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
attr = pd.read_csv('./Data/temp2.csv', names=["Outlook","Temperature","Humidity","Wind","label"])
attr = attr.values
label = attr[:,-1]
attr = np.delete(attr, -1, axis=1)

#quick test for the purity & gain functions
#print("Entropy, Gini, MajorityError")
#for i in range(0,4):
#	print(dt.gain(attr[:,i],label,dt.entropy),"...", dt.gain(attr[:,i],label,dt.gini), "....", dt.gain(attr[:,i],label,dt.majority))

#(self, labels, attr, currlvl, maxd, origs=None, purityfnc=entropy, parent=None)
dec = dt.Tree(label, attr, 1, 8, None, dt.gini)


