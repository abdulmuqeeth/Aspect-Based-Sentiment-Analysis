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
nltk.download('averaged_perceptron_tagger')
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

def BagOfWords(filteredData):
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
	return []

def BeginningWords(x):
	words = []
	for sentence in x:
		if(len(sentence)>=5):
			words.append(sentence[0])
			words.append(sentence[1])
			words.append(sentence[2])
			words.append(sentence[3])
			words.append(sentence[4])
		else:
			for i in range(len(sentence)):
				words.append(sentence[i])
	return words

def EndWords(x):
	words = []
	for sentence in x:

		if(len(sentence)>=5):
			words.append(sentence[-5])
			words.append(sentence[-4])
			words.append(sentence[-3])
			words.append(sentence[-2])
			words.append(sentence[-1])
		else:
			for i in range(len(sentence)):
				words.append(sentence[i])
	return words

def Bigrams(x):
	all_bigrams = []
	final_list = []
	for sentence in x:
		print(nltk.bigrams(sentence))
		all_bigrams.append(list(nltk.bigrams(sentence)))

	for bigram_sentence in all_bigrams:
		for tupl in bigram_sentence:
			final_list.append(tupl[0]+' '+tupl[1])

	print(final_list[0])
	return final_list

def SkipBigrams(x):
	all_skip_bigrams = []
	final_list = []
	for sentence in x:
		all_skip_bigrams.append(nltk.skipgrams(sentence, 2, 5))

	for skipgram_sentence in all_skip_bigrams:
		for tupl in skipgram_sentence:
			final_list.append(tupl[0]+' '+tupl[1])

	return final_list

def SkipBigramsSentence(x):
	all_skip_bigrams = []
	final_list = []
	#print(x)
	#print('xxx')
	all_skip_bigrams.append(list(nltk.skipgrams(x, 2, 5)))
	#print(all_skip_bigrams)
	for tupl in all_skip_bigrams[0]:
		#print('tuple')
		#print(tupl)
		final_list.append(tupl[0]+' '+tupl[1])

	return final_list

def HeadWords(x):
	return []

def WordsPOS(x):
	words = []
	for sentence in x:
		pos_tags = nltk.pos_tag(sentence)
		#print('pos_tags', pos_tags)
		for word in pos_tags:
			#print('word',word)
			# if re.compile('NN*').match(words[1]) is not None:
			if (word[1][0] == 'V' or word[1][0] == 'J' or word[1][0] == 'R'):
				words.append(word[0])
	return words

def Emoticons():
	pos_emoticons = [':)', ': )', ':-)', ';)', ';-)', '=)', '^_^', ':-D', ':-d', ':d', ':D', '=d', '=D', 'C:', 'c:', 'xd', 'XD', 'Xd', 'xD', '(x', '(=', '^^', '^o^', '^O^', '\'u\'', 'n_n', '*_*', '*o*', '*O*', '*_*']
	neg_emoticons = [':-(', ':(', ':((', ': (', 'd:', 'D:', 'Dx', 'n', ':|', '/:', ':\\', '):-/', ':', '=\'[', ':_(', '/T_T', '/t_t' , 'TOT', 'tot', ';_;' ]
	return pos_emoticons + neg_emoticons

def ch_n_gram(x):
	return []

def VerbTags(x):
	pass

def VerbWords(x):
	pass

def Features(x_train, x_train_aspect):
	features = []
	features += WordsAroundVerb(x_train)
	features += EndWords(x_train)
	features += BeginningWords(x_train)
	features += list(x_train_aspect)
	features += Bigrams(x_train)
	features += SkipBigrams(x_train)
	#features += HeadWords(x_train)
	#features += BagOfWords()
	features += WordsPOS(x_train)
	features += Emoticons()
	#features += ch_n_gram(x_train)
	return features

def dict(features, sentence):
	sentence += SkipBigramsSentence(sentence)
	feature_vec = {}
	for feature in features:
		if feature in sentence:
			feature_vec[feature] = 1
		else:
			feature_vec[feature] = 0
	#print(feature_vec)
	#print(sum(feature_vec.values()))
	#print('in dict')
	return feature_vec

def all_documents(format_data, format_labels):
	all_docs = [(format_data[i], format_labels[i]) for i in range(len(format_data))]
	return all_docs

def train_data(tokens, data, labels):
	all_docs = all_documents(data, labels)
	training_data = []
	for document in all_docs:
		# Getting the training data into correct format for nltk.MaxEntClassifier.train
		temp = tuple((dict(tokens, document[0]), document[1]))
		training_data.append(temp)
	return training_data

def test_data(tokens, data, labels):
	all_docs = all_documents(data, labels)
	testing_data = []
	for document in all_docs:
		# Getting the training data into correct format for nltk.MaxEntClassifier.train.classify
		testing_data.append(dict(tokens, document[0]))
	return testing_data


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
	x_train_aspect = np.array(x_train_aspect)

	print('here')
	print(x_train[0])
	print('here')
	print(y_train[0:10])



	features = Features(x_train, x_train_aspect)
	print('printing features')
	print(features)
	print('Length: ',len(features), type(features))

	features = set(features)
	print('Length2: ',len(features))

	# 10-Fold Cross Validation
	kf = KFold(n_splits=10)
	kf.get_n_splits(x_train)

	for train_index, test_index in kf.split(x_train):
		print(type(train_index))
		print(type(x_train))
		errors = 0
		x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
		y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]
		print(type(x_train_aspect))
		x_train_aspect_kf = x_train_aspect[train_index]

		fv = Features(x_train_kf,  x_train_aspect_kf)
		x_train_maxent = train_data(fv, x_train_kf, y_train_kf)
		print('Train feature vectors created')
		
		x_test_maxent = test_data(fv, x_test_kf, y_test_kf)
		print('Test feature vectors created')
		
		mec = MaxentClassifier.train(x_train_maxent)
		print('train finish')

		

		for featureset, label in zip(x_test_maxent, y_test_kf):
			if(mec.classify(featureset) != label):
				errors += 1

		print("Accuracy: %f"% (1 - (errors / float(len(y_test_kf)))))



if __name__  == "__main__":
	main()




