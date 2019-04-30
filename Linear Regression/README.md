# Gradient Descent Information & Usage

------

#### For complete usage, refer to testdriver.py


Two Gradient Descent Techniques are available, Batch Gradient Descent and Stochastic Gradient Descent.
Batch processes each example to update the gradient, Stochastic uses a single randomly chosen example to update the gradient.

### Required Parameters(In order)

-x: an NxC+1 numpy array holding the coefficients of the values. N is the number of examples, C is the number of attributes with 1 added to augment a bias term. 
-y: an Nx1 numpy array holding the output for each example. N is the number of examples
-alphas: an 1xC numpy array which holds the initial starting weights.
-lrnrate: A float corresponding to how big of steps you want to take on updates
-tolerance: A floating point number, which designates when to stop updating according to how much the weights change.
-iterations: An int for max number of iterations to run for

### Return values(In order)

-alphas: a 1xC numpy array which holds the final weight values
-iter: an integer for how many iterations it took to converge

#### Example Usage

```python

#read in our data and separate the data from the output
dframe = pd.read_csv("./Data/temp.csv",names=["x1","x2","x3","Out"])
data = dframe.values
output = data[:,-1]

#Get the col and row count for convenience
colcount = data.shape[1]
rowcount = data.shape[0]


#Get our values and augment our value array.
vals = data[:,0:colcount-1]
output = np.reshape(output, (vals.shape[0],1))
vals = np.concatenate((np.ones([vals.shape[0],1], dtype=np.int),vals),axis=1)

#Initialize alphas 
alphas = np.ones([1,colcount])


#Our hyperparameters
lrnrate = 0.026903
iterations = 40000
tolerance = 0.000001

batalpha, itr = gd.batchGrad(vals,output,alphas,lrnrate,tolerance,iterations)
```