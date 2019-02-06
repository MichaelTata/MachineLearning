import pandas as pd
import numpy as np


def entropy(attribute):	

	##
	##Check on division by zero and log. Just ret 0 for that calc
	##
	num = len(attribute)
	if num == 0:
		return 0
	uniq, counts = np.unique(attribute, return_counts=True)
	val = counts / num
	res = -val.dot(np.log2(val))
	return res


#Calculate the Majority Error to determine data purity
def majority(attribute):
	
	num = len(attribute)
	if num == 0:
		return 0
	
	uniq, counts = np.unique(attribute, return_counts=True)
	
	if counts.size is 1:
		res = 0
	else:
		res = np.amin(counts)/num
	
	return res


#Calculates the Gini Index to determine purity of data
def gini(attribute):

	num = len(attribute)
	uniq, counts = np.unique(attribute, return_counts=True)
	val = counts / num	
	val = np.power(val, 2)
	val = sum(val)
	res = 1 - val
		
	return res

#Calculates the information gain 
def gain(attribute, label, purityfunction):
	
	#Total label count
	total = len(label)
	sum = 0.0
	
	#Get unique values of attr and their corresponding counts
	uniques, counts = np.unique(attribute, return_counts=True)
	num = len(uniques)
	for i in range(0,num):
		
		
		#print(label[attribute==uniques[i]])
		#Take label values for current attribute value
		temp = purityfunction(label[attribute == uniques[i]])
		
		
		sum += counts[i] * temp
					
	return purityfunction(label) - sum / total


##########################


attr = pd.read_csv('./train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
#attr = pd.read_csv('./temp2.csv', names=["Outlook","Temperature","Humidity","Wind","label"])
attr = attr.values
label = attr[:,-1]
attr = np.delete(attr, -1, axis=1)

#quick test for the purity & gain functions
print("Entropy, Gini, MajorityError")
for i in range(0,6):
	print(gain(attr[:,i],label,entropy),"...", gain(attr[:,i],label,gini), "....", gain(attr[:,i],label,majority))






###########################

class Tree:
	label = None
	attribute = None
	choice = None
	children = None
	parent = None
	level = None
	depth = 10
	
	def __init__(self, labels, attr, maxd, parent=None):
	
		if parent is not None:
			self.parent = parent
		
		if maxd is not None:
			self.depth = maxd
		
	def makeSubTree():
				
		return
	def getBest():
		return
	
	