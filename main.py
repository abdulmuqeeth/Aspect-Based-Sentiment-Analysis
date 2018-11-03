import csv
import numpy as np

def main():
	data = []
	with open('data-1_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			data.append(row)
	fields = data[0]
	data = np.array(data[1:])
	print(data.shape, fields)

if __name__  == "__main__":
	main()
