# SVM Information & Usage

------

#### For a class usage example, refer to example.py
#### For each svm type usage example, refer to example.py (for primal SVM using SGD) and example2.py (For dual form SVM with kernels)

To use the svm, initialize an svm object, and then use one of the two techniques(primal, dual(with possible kernel))


### Required Parameters on Initialization(In order)

- PRIMAL vals: an N x C+1 dimensional numpy array. C is the number of attribute columns and the +1 is because we augment our weight vector with our bias, so we have a column of 1's for each row. N is our amount of examples. 
- DUAL vals: an N x C dimensional numpy array. C is the number of attribute columns

- PRIMAL weights: a C+1 length numpy array of ZEROS. C is number of attribute columns, and the +1 is due to our augmented bias term. 
- DUAL weights: a C+1 length numpy array of ZEROS.

-Labels: an Nx1 dimensional numpy array. N is number of examples, and each label value is either -1 or 1


----
Primal Parameters(In order)

-epoch: an integer for how many iterations to run

-learning rate: a float for how big of steps you want to take on weight updates

- Margin Parameter C: Trade off determining how hard of margin you want.

- OPTIONAL learning choice: integer that is 1 or anything else. Determines what learning rate scheduler to use.

---
Dual Parameters(In Order)

- Margin Parameter C: Trade off determining how hard of margin you want.

- OPTIONAL gamma: a float used for the gaussian kernel. Gamma affects the variance. Default gamma is 3.

- OPTIONAL kernelchoice: an integer 0 or 1, 0 for linear, 1 for gaussian. Default 0. anything else OR none for custom. 

- OPTIONAL custom: custom kernel function to be used. Function should take 2 parameters(x1, x2). No error checking made, and no possible extraneous parameters as of now.


### Return Values
 
- For Primal it will return the weights numpy array on completion, which you can use to predict(Refer to examples)

- For Dual, after running dual, call dualpred with the test examples and it will return the prediction numpy array. Simply take the sign of each to get the label(Refer to examples)
- Dual will also return an array whose elements correspond to the support vectors. 

--- Example Creation and usage

```python

#Train data
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"])
vals = vals.values
label = vals[:,-1]

#Change 0 labels to -1
label[label==0]=-1

vals = np.delete(vals, -1, axis=1)
weights = np.zeros(vals.shape[1])


#Test data
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)

#Change 0 labels to -1
testlabel[testlabel==0]=-1


#Margin Parameter
val = 500/873

#Weights
weightsp = np.array(weights)

#Initialize SVM
vecmac = supv.Svm(vals, weightsp, label)

#Dual SVM with linear kernel and 0 gamma(Pointless)
vecmac.dual(val, 0, 0)

#Get 
av, svidx = vecmac.dualpred(testvals)
pred = np.sign(av)

print("\n\n C VAL:", val)
print("\nAccuracy:",((np.sum(pred==testlabel))/testlabel.size))
print("________")




```
