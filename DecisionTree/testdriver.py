import pandas as pd
import numpy as np
import decisiontree as dt



## 0 for categorical attributes, 1 for numerical(or attributes based on ranges)
num = 0

#List of score tuples, where each score tuple corresponds to a tree with max depth of list index + 1, and the 2 scores are for train and
#testing predictions
entscores = []
giscores = []
mescores = []


##Categorical
if num == 0:

	for maxlvl in range(1,17):

		attr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
		#attr = pd.read_csv('./Data/temp2.csv', names=["Outlook","Temperature","Humidity","Wind","label"])

		
		attr = attr.values
		label = attr[:,-1]
		attr = np.delete(attr, -1, axis=1)
		fracs = np.ones_like(label)
		
		decEnt = dt.Tree(label, attr, fracs, 1, maxlvl, None, None, dt.entropy)
		decGi = dt.Tree(label, attr, fracs, 1, maxlvl, purityfnc=dt.gini)
		decErr = dt.Tree(label, attr, fracs, 1, maxlvl, None, None, dt.majority)
		
		
		
		
		
		#read in and split the testing dataset
		testattr = pd.read_csv('./Data/test.csv',names=["buying","maint","doors","persons","lug_boot","safety","label"])
		#testattr = pd.read_csv('./Data/temp2train.csv',names=["Outlook","Temperature","Humidity","Wind","label"])
		
		testattr = testattr.values
		testlabel = testattr[:,-1]
		testattr = np.delete(testattr,-1,axis=1)

		#read in and split the training dataset, for prediction testing.
		trainattr = pd.read_csv('./Data/train.csv', names=["buying","maint","doors","persons","lug_boot","safety","label"])
		
		trainattr = trainattr.values
		trainlabel = trainattr[:,-1]
		trainattr = np.delete(trainattr, -1, axis=1)

		#Get counts of the label values, so we can see how askew the resulting labels may be
		trash,trainlabelcount = np.unique(trainlabel, return_counts=True)
		trash,testlabelcount = np.unique(testlabel, return_counts=True)
		
		predictionstest1 = decEnt.predict(testattr)
		predictionstest2 = decGi.predict(testattr)
		predictionstest3 = decErr.predict(testattr)

		predictionstrain1 = decEnt.predict(trainattr)
		predictionstrain2 = decGi.predict(trainattr)
		predictionstrain3 = decErr.predict(trainattr)
		

		totaltest = np.sum(predictionstest1 == testlabel)
		totaltrain = np.sum(predictionstrain1 == trainlabel)
		print("\n\n\nExample Count - Test:",testlabel.shape[0] , "Train:", trainlabel.shape[0])
		print("\nCorrect Prediction Count for Entropy Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage Entropy - Test:", totaltest/testlabel.shape[0] , " Train:", totaltrain/trainlabel.shape[0])
		
		entscores.append((totaltest/testlabel.shape[0],totaltrain/trainlabel.shape[0]))

		totaltest = np.sum(predictionstest2 == testlabel)
		totaltrain = np.sum(predictionstrain2 == trainlabel)
		print("\nCorrect Prediction Count for GiniIndex Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage Gini - Test:", totaltest/testlabel.shape[0] , " Train:", totaltrain/trainlabel.shape[0])
		
		giscores.append((totaltest/testlabel.shape[0],totaltrain/trainlabel.shape[0]))

		totaltest = np.sum(predictionstest3 == testlabel)
		totaltrain = np.sum(predictionstrain3 == trainlabel)
		print("\nCorrect Prediction Count for Majority Error Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage MajError - Test:", totaltest/testlabel.shape[0] , " Train:", totaltrain/trainlabel.shape[0])
		
		mescores.append((totaltest/testlabel.shape[0],totaltrain/trainlabel.shape[0]))

##Numerical
else:
	
	
	#Count unknown as an attribute(0) or as a missing value(1)
	missing = 1
	
	#numerical attributes indexes
	numerics = np.array([0,5,9,11,12,13,14])
	
	attr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
	attr =  attr.values
	label = attr[:,-1]
	fracs = np.ones_like(label)
	attr = np.delete(attr, -1, axis=1)
	
	
	label[label=="no"] = -1
	label[label=="yes"] = 1

	label = np.array(label, dtype=float)
	#print("\n\nShapes:",attr.shape, "____",label.shape, "_____", fracs.shape)
	
	
	if missing == 1:
		#check every column
		for j in range(0,len(attr[0,:])):
			#find the most common element in this column.
			uniqms, countms = np.unique(attr[:,j], return_counts=True)
			bestindex = np.argmax(countms)
			bestguess = uniqms[bestindex]
			
			#Do not take unknown as most common attribute, as we are trying to replace it
			if bestguess == "unknown":
				countms = np.delete(countms, bestindex)
				bestindex = np.argmax(countms)
				
				
			
				#bestguess = uniqms[bestindex]
				#print("Swag:")
				#print(uniqms)
				#print(countms)	
			#Find unknowns in column, then replace them with fractionals
			matches = (attr[:,j]=="unknown")
				#print(matches)
				#exit()
			subcnt = (matches==True).sum()
				#print(subcnt)
			totms = len(attr[:,0])
			totms = totms-subcnt
			#print("total:", totms)
							#print(attr[:,j])
							#print("BEFORE:")
							#print(attr[matches,j])
			#print(attr)
			#exit()
			for idx, match in enumerate(matches):
				#print("MATCH VAL:", match)
				if match==True:
					temp = 0
					if uniqms[0] == "unknown":
						attr[idx,j] = uniqms[1]
					else:
						attr[idx,j] = uniqms[0]
							
							
					reslbl = label[idx]
					#print(reslbl)
					fracs[idx] = countms[0]/totms
					for idx, val in enumerate(uniqms):
						if idx != 0:
							if val != "unknown":
								cpy = np.array(attr[idx,:])
																		
								cpy[j] = uniqms[idx]
									
								attr = np.append(attr, [cpy], axis=0)
								
								label = np.append(label,reslbl)
									
								fracs = np.append(fracs,countms[idx]/totms)
						
						#print("Nice!")
						#print(attr[-1,:], " with ", label[-1], " and with ", fracs[-1])
		
		
	#for tempo in range(0,len(attr[:,0]), 100):
	#	print(attr[tempo,:], "__", label[tempo], "__", fracs[tempo]) 
	#exit()
	
	print("\n\nShapes:",attr.shape, "____",label.shape, "_____", fracs.shape)
	#print(label)
	
	#Convert numerics to binary categories based off of median 
	for i in range(0,numerics.size):
		median = np.median(attr[:,numerics[i]])
		#print("MEDIAN:",median)
		#print(attr[:,i])
		for x in range(0,len(attr[:,numerics[i]])):
			if attr[x,numerics[i]] < median:
		
				attr[x,numerics[i]] = 0
			else:
				attr[x,numerics[i]] = 1
	
	testattr = pd.read_csv('./Data/Bank/test.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
			
	testattr = testattr.values
	testlabel = testattr[:,-1]
	testattr = np.delete(testattr, -1, axis=1)
		
	#read in and split the training dataset, for prediction testing.
	trainattr = pd.read_csv('./Data/Bank/train.csv', names=["age","job","marital","education","default","balance","housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"])
		
	trainattr = trainattr.values
	trainlabel = trainattr[:,-1]
	trainattr = np.delete(trainattr, -1, axis=1)
		
	#Get counts of the label values, so we can see how askew the resulting labels may be
	trash,trainlabelcount = np.unique(trainlabel, return_counts=True)
	trash,testlabelcount = np.unique(testlabel, return_counts=True)
		
	testlabel[testlabel=="no"] = -1
	testlabel[testlabel=="yes"] = 1
	testlabel = np.array(testlabel, dtype=float)			

	trainlabel[trainlabel=="no"] = -1
	trainlabel[trainlabel=="yes"] = 1
	trainlabel = np.array(trainlabel, dtype=float)	
		
	#Convert to binary attribute based off median(Median of test, NOT train)
	for i in range(0,numerics.size):
		median = np.median(testattr[:,numerics[i]])
		median2 = np.median(trainattr[:,numerics[i]])
			#print("MEDIAN:",median)
			#print(attr[:,i])
		for x in range(0,len(testattr[:,numerics[i]])):
			if testattr[x,numerics[i]] < median:
			
				testattr[x,numerics[i]] = 0
			else:
				testattr[x,numerics[i]] = 1
		for x in range(0,len(trainattr[:,numerics[i]])):
			if trainattr[x,numerics[i]] < median2:
				trainattr[x,numerics[i]] = 0
			else:
				trainattr[x,numerics[i]] = 1
	
	
	for maxlvl in range(1,17):	
		attr1 = np.array(attr)
		label1 = np.array(label, dtype=float)
		fracs1 = np.array(fracs, dtype=float)
		testattr1 = np.array(testattr)
		trainattr1 = np.array(trainattr)
		testlabel1 = np.array(testlabel)
		trainlabel1 = np.array(trainlabel)
		
		
		#print(label.shape)
		
		decEnt = dt.Tree(label1, attr1, fracs, 1, maxlvl, None, None, dt.entropy)
		decGi = dt.Tree(label1, attr1, fracs, 1, maxlvl, None, None, dt.gini)
		decErr = dt.Tree(label1, attr1, fracs, 1, maxlvl, None, None, dt.majority)

		
		#read in and split the testing dataset
		

		
		#Predictions for each type of heuristics, for both the test dataset
		#and train dataset
		predictionstest1 = decEnt.predict(testattr1)
		predictionstest2 = decGi.predict(testattr1)
		predictionstest3 = decErr.predict(testattr1)

		predictionstrain1 = decEnt.predict(trainattr1)
		predictionstrain2 = decGi.predict(trainattr1)
		predictionstrain3 = decErr.predict(trainattr1)
		
		
		
		totaltest = np.sum(predictionstest1 == testlabel1)
		totaltrain = np.sum(predictionstrain1 == testlabel1)
		print("\n\n\nExample Count - Test:",testlabel1.shape[0] , "Train:", trainlabel1.shape[0])
		print("\nCorrect Prediction Count for Entropy Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage Entropy - Test:", totaltest/testlabel1.shape[0] , " Train:", totaltrain/trainlabel1.shape[0])
		
		entscores.append((totaltest/testlabel1.shape[0],totaltrain/trainlabel1.shape[0]))

		
		totaltest = np.sum(predictionstest2 == testlabel1)
		totaltrain = np.sum(predictionstrain2 == testlabel1)
		print("\nCorrect Prediction Count for GiniIndex Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage Gini - Test:", totaltest/testlabel1.shape[0] , " Train:", totaltrain/trainlabel1.shape[0])
		
		giscores.append((totaltest/testlabel1.shape[0],totaltrain/trainlabel1.shape[0]))

		totaltest = np.sum(predictionstest3 == testlabel1)
		totaltrain = np.sum(predictionstrain3 == testlabel1)
		print("\nCorrect Prediction Count for Majority Error Tree - Test:", totaltest, " Train:", totaltrain)
		print("Accuracy Percentage MajError - Test:", totaltest/testlabel1.shape[0] , " Train:", totaltrain/trainlabel1.shape[0])
	
		mescores.append((totaltest/testlabel1.shape[0],totaltrain/trainlabel1.shape[0]))
		


#Output the label counts and then scores of predictions for each tree level.
print("\n_______________________________________________________________________________________")
print("Train Label Counts:", trainlabelcount, " Total:", sum(trainlabelcount))
print("Test Label Counts:", testlabelcount, "  Total:", sum(testlabelcount))

#Averages accross all trees for different heuristics 
avgetest = 0
avgetrain = 0
avggtest = 0
avgmetest = 0
avggtrain = 0
avgmetrain = 0

print("\n-----------------------------------Accuracies------------------------------------------")
print("_______________________________________________________________________________________")
print("Max   |       Entropy         |        GiniIndex         |            MajErr       ")
print("Depth |  Test          Train  |      Test       Train    |        Test           Train")
print("_______________________________________________________________________________________")
for i in range(0,len(entscores)):
	
	if i < 9:
		d = ' '
	else:
		d = ''

	avgetest = avgetest + entscores[i][0]
	avgetrain = avgetrain + entscores[i][1]
	
	avggtest = avggtest + giscores[i][0]
	avggtrain = avggtrain + giscores[i][1]
	
	avgmetest = avgmetest + mescores[i][0]
	avgmetrain = avgmetrain + mescores[i][1]
	
	#print(d+"Lvl"+ str(i+1) +': ','{:.4f}'.format(round(entscores[i][0],4)),"       ",'{:.4f}'.format(round(entscores[i][1],4)))
	print(d+"Lvl"+ str(i+1) +': ','{:.4f}'.format(round(entscores[i][0],4)),"       ",'{:.4f}'.format(round(entscores[i][1],4)),"      ",
		'{:.4f}'.format(round(giscores[i][0],4)),"    ",'{:.4f}'.format(round(giscores[i][1],4)),"         ",'{:.4f}'.format(round(mescores[i][0],4)),"         ",'{:.4f}'.format(round(mescores[i][1],4)))

print("_______________________________________________________________________________________")
print("Avgs:  ", round(avgetest/len(entscores),4), "     ", round(avgetrain/len(entscores),4), "       ", round(avggtest/len(giscores),4),
	"    ",round(avggtrain/len(giscores),4),"         ",round(avgmetest/len(mescores),4),"        ",round(avgmetrain/len(mescores),4))
print("AvgErr:",round(1-avgetest/len(entscores),4), "     ", round(1-avgetrain/len(entscores),4), "       ", round(1-avggtest/len(giscores),4),
	"    ",round(1-avggtrain/len(giscores),4),"         ",round(1-avgmetest/len(mescores),4),"        ",round(1-avgmetrain/len(mescores),4))