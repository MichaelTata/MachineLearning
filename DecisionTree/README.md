NOTE: There is currently no error checking for input to the tree, so make sure you don't pass garbage otherwise it will crash. 

To make and learn a decision tree, all you need to do is initialize an instance of a tree object.
The required parameters for the tree are, in order:


-Labels: an Nx1 dimensional numpy array. N is the number of examples with some label

-Attributes: an NxC dimensional numpy array. C is the number of attribute columns, and N is the amount of examples that MUST align index-wise with their corresponding label.

-Current Level: An integer, just pass 1 for instantiation to designate root node.

-Max depth: An integer, designating how deep you want to allow the tree to grow. If you want to early stop, just adjust this parameter. 

##Non-required parameter(Will default to entropy based information gain)

-purityfnc: The function you want to use to determine how to split the tree using information gain. 
			3 functions are provided: entropy, majority, gini 
			Where majority is the majority error, and gini is the Gini Index.


NOTE: Do not use the other unnamed parameters, they are used for tree recursion and will default to necessary value.
-------------------------------------------------------------------------------------------------------------------------------------------------------

To check the predicted label of the decision tree, you use the created tree objects predict function. For ease of use,
predict finds the predicted labels for an entire numpy array of example attributes at a time. So, the only parameter is:

-Attributes:an NxC dimensional numpy array(Formatted identically to the array that was passed to create the tree)

-------------------------------------------------------------------------------------------------------------------------------------------------------

Example Creation and Prediction usage:

#set max depth of tree
maxlvl = 4

#Reads in the attributes AND labels with pandas package
attr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])

#Convert the pandas dataframe to a numpyarray
attr = attr.values

#Take the label column(Last column in this example) and save it, then strip/delete it from the original attribute array.
label = attr[:,-1]
attr = np.delete(attr, -1, axis=1)

#Make a decision tree based off the Gini Index. 
decGi = dt.Tree(label, attr, 1, maxlvl, purityfnc=dt.gini)


#Read in test attributes and labels with pandas package
testattr = pd.read_csv('./Data/test.csv',names=["buying","maint","doors","persons","lug_boot","safety","label"])

#convert to numpy array and split, just like before. 
testattr = testattr.values
testlabel = testattr[:,-1]
testattr = np.delete(testattr,-1,axis=1)


#Find predicted labels using our decision tree
predictions = decGi.predict(testattr)

#Find and sum examples where the tree correctly predicted the label
totaltest = np.sum(predictions == testlabel)

#Output amount of correctly predicted
print("Percentage of correct predictions:" totaltest/testlabel.shape[0])

-------------------------------------------------------------------------------------------------------------------------------------------------------