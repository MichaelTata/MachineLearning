import pandas as pd
import numpy as np

#Calculates entropy to determine purity
def entropy(uniqcounts):	
	
	if np.any(uniqcounts==0):
		return 0
	
	val = uniqcounts
	
	res = -(val.dot(np.log2(val)))

	return res


#Calculate the Majority Error to determine data purity
#calculates for given attribute value
def majority(uniqcounts):
	
	if len(uniqcounts) == 1:
		return 0
		
	res = np.amin(uniqcounts)
	
	return res


#Calculates the Gini Index to determine purity of data
def gini(uniqcounts):

	val = uniqcounts	

	val = np.power(val, 2)

	val = sum(val)
	
	res = 1 - val
	
	return res

#Calculates the information gain
#based off purity function and summation of subsets
def gain(attribute, label, fracs, purityfunction):
	
	
	
	#Total label count
	total = np.sum(fracs)
	sum = 0.0
	
	#Get unique values of attr and their corresponding counts
	uniques, counts = np.unique(attribute, return_counts=True)
	num = len(uniques)
	totcounts = []
	
	totunq, labeltot = np.unique(label, return_counts=True)
	

	for i in range(0, len(totunq)):
		tvl = np.sum(fracs[label==totunq[i]])
		
		
		
		totcounts.append(tvl/total)
	
	
	
	for i in range(0,num):
		 	
		
		matches = (attribute == uniques[i])
		currlblrows = label[matches]
		
		unqlbl = np.unique((currlblrows))
		curfracrows = fracs[matches]
	
		unqvalcounts = []
		
		
		
		curtot = np.sum(curfracrows)
			
		for s in range(0,len(unqlbl)):
			
			matchfrac = (curfracrows[currlblrows==unqlbl[s]])
			
			temp = np.sum(matchfrac)/curtot
			
					
			unqvalcounts.append(temp)		
		
		#Take label values for current attribute value
		pval = purityfunction(np.array(unqvalcounts))
		
		sum += (curtot/total) * pval
		
	
	totsum = purityfunction(np.array(totcounts))
	
	#expsm = sum / total
	
	
	#Return abs in case of floating point problems(can go into negative 0)
	return abs(totsum-sum) 

#Decision Tree based off of the ID3 algorithm
class Tree:
	label = None     #(string/int/fp)Resulting label if any.
	attribute = None #(int index)True attribute column that this tree split on
	choice = None    #(int index)Choice of best attribute split. The value the attribute takes on
	children = None  #(list of Tree objects) children branches for this tree
	parent = None    #(tree object) parent tree
	level = None     #(int) current level
	maxdepth = 16    #(int) max level growable
	originals = None #(numpy array) Original Columns preserved, used to keep track of
					 #original attributes after splitting
	#lbltype=None 	 #label dtype for numpy array. If we are using Adaboost, float or int are needed
					 #Otherwise object is fine. 
		
	#labels    - numpy array which holds the resulting label for each example
	#
	#attr      - numpy array which holds the examples and all its attributes
	#
	#fracs 	   - numpy array which holds the fractional weight(or 1.0 if it is a complete example)
	#
	#currlvl   - integer describing level this tree is at with 1 being the root
	#
	#maxd      - integer to bound how big the tree can grow(its maximum depth)
	#
	#origs     - numpy array which holds original column orderings
	#
	#purityfnc - Function to calculate purity of attribute data.(3 heuristics given here:entropy,
	#			 gini index, and majority)
	#
	#parent    - parent tree object
	#
	def __init__(self, labels, attr, fracs, currlvl, maxd=17, pchoice=None, origs=None, purityfnc=entropy, parent=None):
		
		#create children list and the current level
		self.children = []
		self.level = currlvl
		self.lbltype = labels.dtype
	
		#Set parent
		if parent is not None:
			self.parent = parent
		
		#Set Max depth.
		if maxd is not None:
			self.maxdepth = maxd
		
		#Set Choice, or attribute value resulting from split
		if pchoice is not None:
			self.choice = pchoice
		
		#Save original column layouts.(Rows get deleted due to splitting, so originals changes
		#on each subtree)
		if origs is None:
			self.originals = np.array(attr, copy=True)
		else:
			self.originals = np.array(origs, copy=True)
		
		#Edge case of a single label and example remaining
		if np.isscalar(labels):	
			self.label = labels	
			#print("RES:",self.label)
			return 
		
		#Decision Stump Wanted, so make leaf nodes for best attr split and finish. 
		if maxd == 1 and parent is None:
			self.makeSubTree(labels, attr, fracs, purityfnc)
			return

		
		#If we have nothing left or we cannot grow, return most common label, also includes cases where I split best attr and it was last attr
		#in that case, just take best label.
		if np.size(attr, 0) == 0 or attr.size==0 or self.level >= self.maxdepth:
			uniq, count = np.unique(labels, return_counts=True)
			curr = 0
			max = 0
			
			for idx, val in enumerate(uniq):
				curr = np.sum(fracs[labels==uniq[idx]])
				if curr >= max:
					max = curr
					best = val
					
			self.label = best		
			#print("RES:",self.label)
			return 
		
		#All labels are the same so we have our leaf node
		if np.all(labels == labels[0]):
			self.label = labels[0]
			#print("RES:",self.label)
			return
		
		#Begin recursion(Building subtrees based off best split)
		self.makeSubTree(labels, attr, fracs, purityfnc)	
		return
	
		
	#GetBest finds the best attribute to split on based on the purity function used.
	#Requires the labels along with the attributes, to decide which attribute is most pure
	#Purity is the heuristic you want to use(Gini, Entropy, MajError...)
	def getBest(self, labels, attr, fracs, purity):
		
		#Get size of columns to check
		colsz = attr.shape[1]
		
		gains = np.empty([colsz])
				
		for i in range(colsz):
			
			#Get all examples of attribute at current index
			#Then pass to find the information gain
			pattr = attr[:,i]
			gains[i] = gain(pattr, labels, fracs, purity)
			
			#print("GAINS:", gains[i])
		
		#print("GAINS:",gains)
		return np.argmax(gains)
	
	
	
	#Recursive Function that gets the best attribute to split on
	#then will make subtrees for each value that the best attribute can take on
	#
	def makeSubTree(self, labels, attr, fracs, purity):
		#Find the best attribute to split on
		
		
		currbest = self.getBest(labels, attr, fracs, purity)
		bestcol = attr[:,currbest]
		
	
		#Compares column to original columns to find original attribute.
		#Should probably have used a dictionary or mapping of some sort
		#But numpy makes this very easy to quickly implement
		for i in range(0, np.size(self.originals,1)):		
			if np.array_equal(bestcol,self.originals[:,i]):				
				self.attribute = i
				
		
		#cut the best column from all examples
		cutattr = np.delete(attr,[currbest],1)

		
		#Create subtrees for the best splitting attribute
		#Creating a branch for each unique value of this choice
		for uniq in np.unique(bestcol):
			
			#Get the rows for this value
			childorig = self.originals[bestcol==uniq,:]
			childlabels = labels[bestcol == uniq]
			childfracs = fracs[bestcol == uniq]
			splitrows = attr[bestcol==uniq,:]
			
			#split the best column from the rows for this choice
			childattr = np.delete(splitrows,[currbest],1)	
					
			#Make the subtree(Recurse) for this choice and value, adding the subtree as a child on its completion
			child = Tree(childlabels, childattr, childfracs,self.level+1,self.maxdepth, uniq, childorig,purity,self)
			self.children.append(child)		
		return
	
	
	#Given new examples(attr), use the decision tree to predict what the resulting label
	#will be.
	#Returns array of predictions of the same length as the examples array(attr) passed in
	def predict(self, attr):
		#Create empty array to store all labels
		temp = np.ones(attr.shape[0], dtype=self.lbltype)
			
		#Check if we are at a leaf node and if so take the label
		if len(self.children) == 0:
			predictions = temp
			for i in range(0,np.size(predictions)):

				predictions[i] = self.label
				
			return predictions
	
		predictions = temp
		
		#Go through each child where the given example matches our 
		#
		for child in self.children:
			
			matches = (attr[:,self.attribute]==child.choice) 
	
			chvals = child.predict(attr[matches])
			predictions[matches] = chvals
				
	
		return predictions
	
	
	
	
	

	
#Decision Tree based off of the ID3 algorithm, uses random subset of features to determine split, Speceifically for bagging and random forest algorithms.
class RandTree:
	label = None     #(string/int/fp)Resulting label if any.
	attribute = None #(int index)True attribute column that this tree split on
	choice = None    #(int index)Choice of best attribute split. The value the attribute takes on
	children = None  #(list of Tree objects) children branches for this tree
	parent = None    #(tree object) parent tree
	level = None     #(int) current level
	maxdepth = 16    #(int) max level growable
	originals = None #(numpy array) Original Columns preserved, used to keep track of
					 #original attributes after splitting
	#lbltype=None 	 #label dtype for numpy array. If we are using Adaboost, float or int are needed
					 #Otherwise object is fine. 
		
	#labels    - numpy array which holds the resulting label for each example
	#
	#attr      - numpy array which holds the examples and all its attributes
	#
	#fracs 	   - numpy array which holds the fractional weight(or 1.0 if it is a complete example)
	#
	#currlvl   - integer describing level this tree is at with 1 being the root
	#
	#maxd      - integer to bound how big the tree can grow(its maximum depth)
	#
	#origs     - numpy array which holds original column orderings
	#
	#purityfnc - Function to calculate purity of attribute data.(3 heuristics given here:entropy,
	#			 gini index, and majority)
	#
	#parent    - parent tree object
	#
	def __init__(self, labels, attr, fracs, currlvl, maxd=17, pchoice=None, origs=None, purityfnc=entropy, parent=None):
		
		#create children list and the current level
		self.children = []
		self.level = currlvl
		self.lbltype = labels.dtype
	
		#Set parent
		if parent is not None:
			self.parent = parent
		
		#Set Max depth.
		if maxd is not None:
			self.maxdepth = maxd
		
		#Set Choice, or attribute value resulting from split
		if pchoice is not None:
			self.choice = pchoice
		
		#Save original column layouts.(Rows get deleted due to splitting, so originals changes
		#on each subtree)
		if origs is None:
			self.originals = np.array(attr, copy=True)
		else:
			self.originals = np.array(origs, copy=True)
		
		#Edge case of a single label and example remaining
		if np.isscalar(labels):	
			self.label = labels		
			return 
		
		#Decision Stump Wanted, so make leaf nodes for best attr split and finish. 
		if maxd == 1 and parent is None:
			self.makeSubTree(labels, attr, fracs, purityfnc)
			return

		
		#If we have nothing left or we cannot grow, return most common label, also includes cases where I split best attr and it was last attr
		#in that case, just take best label.
		if np.size(attr, 0) == 0 or attr.size==0 or self.level >= self.maxdepth:
			uniq, count = np.unique(labels, return_counts=True)
			curr = 0
			max = 0
			
			for idx, val in enumerate(uniq):
				curr = np.sum(fracs[labels==uniq[idx]])
				if curr >= max:
					max = curr
					best = val
					
			self.label = best		
			#print("RES:",self.label)
			return 
		
		#All labels are the same so we have our leaf node
		if np.all(labels == labels[0]):
			self.label = labels[0]
			#print("RES:",self.label)
			return
		
		#Begin recursion(Building subtrees based off best split)
		self.makeSubTree(labels, attr, fracs, purityfnc)	
		return
	
		
	#GetBest finds the best attribute to split on based on the purity function used.
	#Requires the labels along with the attributes, to decide which attribute is most pure
	#Purity is the heuristic you want to use(Gini, Entropy, MajError...)
	def getBest(self, labels, attr, fracs, subset, purity):
		
		#Get size of columns to check
		colsz = attr.shape[1]
		
		gains = np.empty([colsz])
				
		for i in range(colsz):
			
			if i in subset:
				#Get all examples of attribute at current index, and is in random samples
				#Then pass to find the information gain
				pattr = attr[:,i]
				gains[i] = gain(pattr, labels, fracs, purity)
			else:
				gains[i] = 0
			#print("GAINS:", gains[i])
		
		#print("GAINS:",gains)
		return np.argmax(gains)
	
	
	
	#Recursive Function that gets the best attribute to split on
	#then will make subtrees for each value that the best attribute can take on
	#
	def makeSubTree(self, labels, attr, fracs, purity):
		#Find the best attribute to split on
		
		
		##
		## Create random subset here, and pass to current best, and that is literally it. Holy shit
		##
		##
		##
		subset = np.random.choice(attr.shape[1], 4, replace=False)
		currbest = self.getBest(labels, attr, fracs, subset, purity)
		bestcol = attr[:,currbest]
		
		
		
		#Compares column to original columns to find original attribute.
		#Should probably have used a dictionary or mapping of some sort
		#But numpy makes this very easy to quickly implement
		for i in range(0, np.size(self.originals,1)):		
			if np.array_equal(bestcol,self.originals[:,i]):				
				self.attribute = i
				
		
		#cut the best column from all examples
		cutattr = np.delete(attr,[currbest],1)

		
		#Create subtrees for the best splitting attribute
		#Creating a branch for each unique value of this choice
		for uniq in np.unique(bestcol):
			
			#Get the rows for this value
			childorig = self.originals[bestcol==uniq,:]
			childlabels = labels[bestcol == uniq]
			childfracs = fracs[bestcol == uniq]
			splitrows = attr[bestcol==uniq,:]
			
			#split the best column from the rows for this choice
			childattr = np.delete(splitrows,[currbest],1)	
					
			#Make the subtree(Recurse) for this choice and value, adding the subtree as a child on its completion
			child = RandTree(childlabels, childattr, childfracs,self.level+1,self.maxdepth, uniq, childorig,self.rand,purity,self)
			self.children.append(child)		
		return
	
	
	#Given new examples(attr), use the decision tree to predict what the resulting label
	#will be.
	#Returns array of predictions of the same length as the examples array(attr) passed in
	def predict(self, attr):
		#Create empty array to store all labels
		temp = np.ones(attr.shape[0], dtype=self.lbltype)
			
		#Check if we are at a leaf node and if so take the label
		if len(self.children) == 0:
			predictions = temp
			for i in range(0,np.size(predictions)):

				predictions[i] = self.label
				
			return predictions
	
		predictions = temp
		
		#Go through each child where the given example matches our 
		#
		for child in self.children:
			
			matches = (attr[:,self.attribute]==child.choice) 
	
			chvals = child.predict(attr[matches])
			predictions[matches] = chvals
				
	
		return predictions
	
	
	
		
	
	
	