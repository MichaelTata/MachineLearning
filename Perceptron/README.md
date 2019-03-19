# Perceptron Information & Usage

------

#### For a class usage example, refer to example.py
#### For each perceptron type usage example, refer to functionusage.py

To use the perceptron, initialize a perceptron object, and then use one of the three techniques(averaged, standard, voting)


### Required Parameters(In order)

-vals: an N x C+1 dimensional numpy array. C is the number of attribute columns and the +1 is because we augment our weight vector with our bias, so we have a column of 1's for each row. N is our amount of examples. 

-weights: a C+1 length numpy array of ZEROS. C is number of attribute columns, and the +1 is due to our augmented bias term. 

-Labels: an Nx1 dimensional numpy array. N is number of examples, and each label value is either -1 or 1

-epoch: an integer for how many iterations to run

-learning rate: a float for how big of steps you want to take on weight updates


### Return Values
 
 Standard & Averaged:
 Both return their weight vectors(np array), followed by their history(A list of previous weights, as np arrays)
 
 Voting:
 Returns allw(a list of all weights from each update), followed by its corresponding vote count(An np array where each index corresponds to the weight vector in allw)




--- Example Creation and usage

```

#read in train values and labels with pandas.
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"])
vals = vals.values
label = vals[:,-1]

#delete the last column from values as it is our label values
vals = np.delete(vals, -1, axis=1)

#Augment our value matrix with a column of ones
vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)


#Do the same thing for our test values
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)
testvals = np.concatenate((np.ones([testvals.shape[0],1], dtype=np.int),testvals),axis=1)

#create 0 initialized weight vector
weights = np.zeros(vals.shape[1])

#set hyperparameters to pass to perceptron
epochs = 10
lrnrate = 0.2
tolerance = 0.05

#Create our new perceptron object with the designated values
weightsp = np.array(weights)
ptron = pr.perceptron(vals, weightsp, label, epochs, lrnrate)

#Find the standard preceptrons Weight vector
stw, stdhist = ptron.standard()
standard = np.sign(np.dot(stw, testvals.T))


#Print the resulting weight vector and its prediction accuracy
print("Standard Final Weights:", stw)
print("STANDARD MISSES:", testlabel.size - np.sum(standard==testlabel), "   Accuracy:", ((np.sum(standard==testlabel))/testlabel.size))



```
