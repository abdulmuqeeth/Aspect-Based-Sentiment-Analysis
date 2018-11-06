import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def textFilter(tokenizedText):

	stop_words = set(stopwords.words('english'))\
	stop_words.add('[')
	stop_words.add(']')
	filteredText = []

	for word in tokenizedText:
		if word not in stop_words:
			#keep or remove?
			if(word == 'comma'):
				filteredText.append(',')
			else:
				filteredText.append(word)
	
	return filteredText

def filterData(allData):
	filteredData = np.deepcopy(allData)
	
	for i in range (len(data)):
		text = filteredData[i][1]
		tokenizedText = word_tokenize(text)
		filteredData[i][1] = textFilter(tokenizedText)
	
	return filteredData

def bagOfWords(filteredData):
	bow = []
	for i in range(len(filteredData)):
		words = filteredData[i][1]
		bow.extend(words)	
	#print(len(bow))

	lower = lambda x:x.lower(), bow
	bow = set(bow)
	#print(len(bow))

	return bow


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




