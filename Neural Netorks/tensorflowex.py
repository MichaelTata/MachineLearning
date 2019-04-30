import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from keras import layers
from keras import initializers 




def nnmodel( vals, labels, tvals, tlabs, neurons=10, layercount=5, epoc=5, activ='tanh', batchsz=16, dpo=None, dorate=0.2):
		
	#Counts of examples and features
	examples = vals.shape[0]
	features = vals.shape[1]
		
	model = models.Sequential()
		
	for i in range(layercount):
		
		#Current set up is only use one activation function for all layers(Not ideal.)
		if activ == 'tanh':
			#Xavier initialization with tanh activation
			model.add(layers.Dense(neurons, input_dim=features, activation=activ, kernel_initializer='glorot_normal', bias_initializer='zeros', use_bias=True)) 
				
		elif activ == 'relu':
			#he_normal initialization with relu activation
			model.add(layers.Dense(neurons, input_dim=features, activation=activ, kernel_initializer='he_normal', bias_initializer='zeros', use_bias=True)) #He 
		
		else:
			model.add(layers.Dense(neurons, input_dim=features, activation=activ, use_bias=True))
		
		
		if dpo is not None:
			#dropout layers
			model.add(layers.Dropout(dorate))
		
	#output layer
	model.add(layers.Dense(1, activation=activ))	
		
		
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	
	history = model.fit(vals, labels, epochs=epoc, batch_size = batchsz, validation_data=(tvals,tlabs), verbose=0, shuffle=True)
	
	
	
	return history
	



#Train data
vals = pd.read_csv("./Data/bank-note/train.csv",names=["x1","x2","x3","x4","y"]) 
vals = vals.values 
label = vals[:,-1] 
vals = np.delete(vals, -1, axis=1) 

#Test data
testvals = pd.read_csv("./Data/bank-note/test.csv",names=["x1","x2","x3","x4","y"])
testvals = testvals.values
testlabel = testvals[:,-1]
testvals = np.delete(testvals, -1, axis=1)

#Reshape labels from (labelcount,) to (labelcount,1)
label = np.reshape(label, (label.size,1))  			   
testlabel = np.reshape(testlabel, (testlabel.size,1))  

depths = [3,5,9]
widths = [5,10,25,50,100]

histtan = []
histrelu = []


#epochs for model training
epo = 25	

for depth in depths:       #layer count
	
	for width in widths:   #neuron count in layer
		
		#tanh activation model
		histtan.append((nnmodel(vals, label, testvals, testlabel, neurons=width, layercount=depth, activ='tanh', epoc=epo), depth,width))
		
		#relu activation model
		histrelu.append((nnmodel(vals, label, testvals, testlabel, neurons=width, layercount=depth, activ='relu', epoc=epo),depth, width))


#print results.
for i in range(len(histtan)):
	print("\nTan Activ: \t DEPTH:",histtan[i][1] ," \tWIDTH:", histtan[i][2], "\tAcc:", histtan[i][0].history['val_acc'][-1])
	print("Relu Activ:  \t DEPTH:",histrelu[i][1] ," \tWIDTH:", histrelu[i][2], "\tAcc:", histrelu[i][0].history['val_acc'][-1])



#write training and validation results to file(For every model)

#with open("fithistory.txt", 'w') as f:	
#	for i in range(len(histtan)):
#		print("\nTan Activ: \t DEPTH:",histtan[i][1] ," \tWIDTH:", histtan[i][2], "\tVal Acc:", histtan[i][0].history['val_acc'][-1])
#		print("Relu Activ:  \t DEPTH:",histrelu[i][1] ," \tWIDTH:", histrelu[i][2], "\tVal Acc:", histrelu[i][0].history['val_acc'][-1])
#		f.write('\n\nTan Activation   Depth:%s   Width:%s\n' % (histtan[i][1], histtan[i][2]))
#		#for key,value in histtan[i][0].history.items():
#		#	f.write('%s:%s\n' % (key, value))
#		f.write('val_acc:%s\n' % histtan[i][0].history['val_acc'][-1])
#			
#		f.write('RELU Activation   Depth:%s   Width:%s\n' % (histrelu[i][1], histrelu[i][2]))
#		#for key,value in histrelu[i][0].history.items():
#		#	f.write('%s:%s\n' % (key, value))	
#		f.write('val_acc:%s\n' % histrelu[i][0].history['val_acc'][-1])
		
		
		
