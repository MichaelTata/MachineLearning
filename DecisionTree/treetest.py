import csv

#Load in training data
csvfile = open('./train.csv')
reader = csv.DictReader(csvfile, fieldnames=("buying","maint","doors","persons","lug_boot","safety","label"))

for row in reader:
	#process each row 
	print(row['label'], row['buying'])
	

	