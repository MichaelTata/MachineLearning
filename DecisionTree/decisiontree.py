import pandas as pd
import numpy as np

#
def entropy(attribute):	

	##
	##Check on division by zero and log. Just ret 0 for that calc
	##
	num = len(attribute)
	if num == 0:
		return 0
	uniq, counts = np.unique(attribute, return_counts=True)
	val = counts / num
	
	#val = np.ma.masked_array(val, mask=(val<=0), fill_value=0)
	#print("Original val:", val , "- Counts:", counts, "- Num:", num)
	
	res = -val.dot(np.log2(val))
	
	#print("Prev Res", res)
	#res = np.ma.filled(res)
	
	#print("Res:", res)
	return res


#Calculate the Majority Error to determine data purity
#calculates for given attribute value
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

#Decision Tree based off of the ID3 algorithm
class Tree:
	label = None     #(string/int/fp)Resulting label if any.
	attribute = None #(int index)True attribute column that this tree split on
	choice = None    #(int index)Choice of best attribute split. May be unneeded.
	children = None  #(list of Tree objects) children branches for this tree
	parent = None    #(tree object) parent tree
	level = None     #(int) current level
	maxdepth = 6     #(int) max level growable
	originals = None #(numpy array) Original Columns preserved, used to keep track of
					 #original attributes after splitting
	
	#labels    - numpy array which holds the resulting label for each example
	#
	#attr      - numpy array which holds the examples and all its attributes
	#
	#currlvl   - integer describing level this tree is at with 1 being the root
	#
	#maxd      - integer to bound how big the tree can grow(its maximum depth)
	#
	#origs     - numpy array which holds original column orderings
	#
	#purityfnc - Function to calculate purity for information gain.(3 are given here:entropy,
	#			 gini index, and majority)
	#
	#parent    - parent tree object
	#
	def __init__(self, labels, attr, currlvl, maxd, pchoice=None, origs=None, purityfnc=entropy, parent=None):
		
		print("\nNew Tree")
		
		self.children = []
		self.level = currlvl
		print("Level:", self.level)
		
		if parent is not None:
			self.parent = parent
		
		if maxd is not None:
			self.maxdepth = maxd
			
		if pchoice is not None:
			self.choice = pchoice
		
		#Save original column layouts.(Rows get deleted due to splitting, so originals changes
		#on each subtree)
		if origs is None:
			self.originals = np.array(attr, copy=True)
		else:
			self.originals = np.array(origs, copy=True)

		#If we have nothing left or we cannot grow, return most common label
		if np.size(attr, 0) == 0 or self.level == self.maxdepth:
			uniq, count = np.unique(labels, return_counts=True)
			idx = np.argmax(count)		
			#print("\tCount:", count[idx],"\tResLabel:", uniq[idx])
			self.label = uniq[idx]
			print("result:", self.label)
			return 
		
		#All labels are the same so we have our leaf node
		if np.all(labels == labels[0]):
			self.label = labels[0]
			
			print("result:", self.label)
			return
		
		#Begin recursion(Building subtrees based off best split)
		self.makeSubTree(labels, attr, purityfnc)	
	
	
	
	#
	#
	#
	def getBest(self, labels, attr, purity):
		#Get size of columns to check
		colsz = np.size(attr, 1)
		gains = np.empty([colsz])
				
		for i in range(0,colsz):
			#Get all examples of attribute at current index
			#Then pass to find the information gain
			pattr = attr[:,i]
			gains[i] = gain(pattr, labels, purity)
		
		
		
		#print("Gains:", gains)	
		#print("With Max At Column:", np.argmax(gains))
		return np.argmax(gains)
	
	
	
	#Recursive Function that gets the best attribute to split on
	#then will make subtrees for each value that the best attribute can take on
	#
	def makeSubTree(self, labels, attr, purity):
		#Find the best attribute to split on
		currbest = self.getBest(labels, attr, purity)	
		bestcol = attr[:,currbest]
		
		#Compares column to original columns to find original attribute.
		#Should probably have used a dictionary or mapping of some sort
		#But numpy makes this very easy to quickly implement
		for i in range(0, np.size(self.originals,1)):		
			if np.array_equal(bestcol,self.originals[:,i]):				
				self.attribute = i
				print("Attribute Choice:" ,self.attribute)
		
		#cut the best column from all examples
		cutattr = np.delete(attr,[currbest],1)
		
		#Create subtrees for the best splitting attribute
		#Creating a branch for each unique value of this choice
		for uniq in np.unique(bestcol):
			
			#Get the rows for this value
			childorig = self.originals[bestcol==uniq,:]
			childlabels = labels[bestcol == uniq]
			splitrows = attr[bestcol==uniq,:]
			
			#split the best column from the rows for this choice
			childattr = np.delete(splitrows,[currbest],1)	
					
			#Make the subtree(Recurse) for this choice and value, adding the subtree as a child on its return
			child = Tree(childlabels, childattr,self.level+1,self.maxdepth, uniq, childorig,purity,self)
			self.children.append(child)		
		return
	
	
	#Given new examples(attr), use the decision tree to predict what the resulting label
	#will be
	#
	def predict(self, attr):
	
		#print(attr)
		
		temp = np.empty(len(attr), dtype=object)
		
		#print(attr)
		#print("LENGTH:" , len(temp))
		#print(attr)
		
		if len(self.children) == 0:
			#print("RESULTING LABEL!:", self.label)
			predictions = temp
			for i in range(0,len(predictions)):				
				predictions[i] = self.label
				#print("AFTER:",predictions[i])
			#print("PREDICTIONS :", predictions)
			return predictions
	
		predictions = temp
		
		#print("LENGTH:", attr.size)
		for child in self.children:
			#print("Attr path:" , child.attribute, "Child Attr Choice: ", child.choice)
			#print("CURR:", self.attribute) 
			matches = (attr[:,self.attribute]==child.choice) 
			#matches = attr[:,self.attribute]
			#print("MATCH: ", matches)
			#matches = matches==child.choice
			#print("NEWMATCHES: ", matches)
			#matches = np.where(attr[:,self.attribute]==child.choice)
			#print(matches)
			predictions[matches] = child.predict(attr[matches])
	
		return predictions
	
	
	
	
	
	
	