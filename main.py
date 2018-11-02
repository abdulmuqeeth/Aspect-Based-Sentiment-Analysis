import csv
import numpy as np

def main():
	data = []
	with open('data-1_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			data.append(row)
	data = np.array(data)
	print(data.shape)

if __name__  == "__main__":
	main()
