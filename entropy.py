import csv
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import *
from nltk import MaxentClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download()

def textFilter(tokenizedText):

	stop_words = set(stopwords.words('english'))
	stop_words.add('[')
	stop_words.add(']')
	filteredText = []

	for word in tokenizedText:
		if word not in stop_words:
			#keep or remove?
			if(word == 'comma'):
				filteredText.append(',')
			else:
				word = str(PorterStemmer().stem(word))
				filteredText.append(word)
	
	return filteredText

def filterData(allData):
	filteredData = np.copy(allData)
	
	for i in range (len(allData)):
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
	bow = np.array(list(bow))

	return bow

def WordsAroundVerb(x):
	pass

def BeginningWords(x):
	words = []
	for sentence in x:
		words.append(sentence[:5])
	return words

def EndWords(x):
	words = []
	for sentence in x:
		words.append(sentence[5:])
	return words

def Bigrams(x):
	all_bigrams = []
	for sentence in x:
		all_bigrams.append(list(nltk.bigrams(sentence)))
	return all_bigrams

def SkipBigrams(x):
	all_skip_bigrams = []
	for sentence in x:
		all_skip_bigrams.append(nltk.util.skipgrams(sentence, 2, 2))
		all_skip_bigrams.append(nltk.util.skipgrams(sentence, 2, 3))
		all_skip_bigrams.append(nltk.util.skipgrams(sentence, 2, 4))
		all_skip_bigrams.append(nltk.util.skipgrams(sentence, 2, 5))
	return all_skip_bigrams

def HeadWords(x):
	pass

def WordsPOS(x):
	words = []
	for sentence in x:
		pos_tags = nltk.pos_tagger(sentence)
		for word in pos_tags:
			# if re.compile('NN*').match(words[1]) is not None:
			if words[1][0] == 'V' or words[1][0] == 'J' or words[1][0] == 'R'
				words.append(word[0])
	return words

def Emoticons(x):
	pos_emoticons = [':)', ': )', ':-)', ';)', ';-)', '=)', '^_^', ':-D', ':-d', ':d', ':D', '=d', '=D', 'C:', 'c:', 'xd', 'XD', 'Xd', 'xD', '(x', '(=', '^^', '^o^', '^O^', '\'u\'', 'n_n', '*_*', '*o*', '*O*', '*_*']
	neg_emoticons = [':-(', ':(', ':((', ': (', 'd:', 'D:', 'Dx', 'n', ':|', '/:', ':\\', '):-/', ':', '=\'[', ':_(', '/T_T', '/t_t' , 'TOT', 'tot', ';_;' ]
	return pos_emoticons + neg_emoticons

def ch_n_gram(x):
	pass

def VerbTags(x):
	pass

def VerbWords(x):
	pass

def main():
	data = []
	with open('data-1_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			data.append(row)
	fields = data[0]
	data = np.array(data[1:], dtype=object)
	print(data.shape, fields)
	words = filterData(data)
	print(words.shape)

	x_train = []
	y_train = []
	x_train_aspect = []
	for i in range(len(data)):
		x_train.append(words[i][1])
		y_train.append(data[i][4])
		x_train_aspect.append(data[i][2])

	x_train = np.array(x_train)
	y_train = np.array(y_train)

	# 10-Fold Cross Validation
	kf = KFold(n_splits=10)
	kf.get_n_splits(x_train)

	for train_index, test_index in kf.split(x_train):
		x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
		y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

		vectorizer = TfidfVectorizer()
		print(x_train_kf)
		tfidf_x_train_kf = vectorizer.fit(x_train_kf)

		print(vectorizer.get_feature_names())

		mec_train_data = list(zip(onehot_xtrain_kf, y_train_kf))
		mec = MaxentClassifier.train(mec_train_data)



if __name__  == "__main__":
	main()




