import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import random
# nltk.download()
a = []
b = []
b = set(b)


def SkipBigrams(tokenized_sentence):
	all_skip_bigrams = []
	final_list = []
	
	all_skip_bigrams.append(nltk.skipgrams(tokenized_sentence, 2, 0))

	#print(all_skip_bigrams[0])
	#print(list(all_skip_bigrams[0]))
	
	for tupl in list(all_skip_bigrams[0]):
		final_list.append(tupl[0]+' '+tupl[1])

	return final_list

def SkipTrigrams(tokenized_sentence):
	all_skip_bigrams = []
	final_list = []
	
	all_skip_bigrams.append(nltk.skipgrams(tokenized_sentence, 3, 0))

	#print(all_skip_bigrams[0])
	#print(list(all_skip_bigrams[0]))
	
	for tupl in list(all_skip_bigrams[0]):
		final_list.append(tupl[0]+' '+tupl[1]+' '+tupl[2])

	return final_list

def textFilter(tokenizedText):
	global a
	global b
	new_tokenized = []
	comma = '[comma]'

	for word in tokenizedText:
		if(word.find(comma) != -1):
			location = word.find(comma)
			new_word = word[:location]
			new_tokenized.append(new_word)
			#new_tokenized.append(',')
		else:
			new_tokenized.append(word)

	#stop_words = set(stopwords.words('english'))
	stop_words = []
	filteredText = []

	for word in new_tokenized:
		if word not in stop_words:
			#if(word == ':)'):
				#print('#######FOUND#########')
			word = str(PorterStemmer().stem(word))
			
			filteredText.append(word)

	bigrams = SkipBigrams(filteredText)
	trigrams = SkipTrigrams(filteredText)
	filteredText = filteredText + bigrams + trigrams

	for word in filteredText:
		a.append(word)
		b.add(word)
	
	return filteredText

def tokenize_words(text):
	return text.split()

def filterData(allData):

	print('in filtered data')

	filteredData = np.copy(allData)
	
	for i in range (len(allData)):

		print(i)

		text = filteredData[i][1]
		
		aspect_term = filteredData[i][2]
		#aspect_location = filteredData[i][3]
		aspect_location = text.find(aspect_term)


		print('printing text')
		print(text)
		print('Printing aspect term')
		print(aspect_term)
		#print('printing location')
		#print(aspect_location)

		j=0

		trimmed_text = text
		

		for z in range(aspect_location+len(aspect_term), len(text)):
			if(text[z] == ' '):
				j += 1
			if(j==5):
				trimmed_text = text[:z]
				break

		print('Trimmed text')
		print(trimmed_text)
		new_trimmed = trimmed_text

		#Removing Words before the aspect terms
		j=0 
		for z in range(aspect_location, 0, -1):
			if(trimmed_text[z] == ' '):
				j += 1
			if(j==5):
				new_trimmed = trimmed_text[z:]
				break

		#print('##new trimmed text##')
		#print(new_trimmed)


		#tokenizedText = word_tokenize(text)
		tokenizedText = tokenize_words(new_trimmed)
		print('tokenized text')
		print(list(tokenizedText))
		filteredText = textFilter(tokenizedText)
		filteredData[i][1] = filteredText
		print('Filtered Text')
		print(filteredText)
		# if i==9:
		#  	break
	
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

def label2bool(labels):
    nn_labels = []
    for label in labels:
    	if label == '-1':
    		nn_labels.append([1,0,0])
    	if label == '0':
    		nn_labels.append([0,1,0])
    	if label == '1':
    		nn_labels.append([0,0,1])
    return np.array(nn_labels)
 
def get_batch(X, y, batch_size):
	for batch_pos in range(0,len(X),batch_size):
		yield X[batch_pos:batch_pos+batch_size], y[batch_pos:batch_pos+batch_size] 

def encode_onehot(sentence, tokens):
	encoded_data = [0 for i in range(len(tokens))]
	for word in sentence:
		if word in tokens:
			j = tokens.index(word)
			encoded_data[j] = 1
	return np.array(encoded_data)

def random_undersampler(all_data):

	new_data = all_data.copy()
	p = 0
	while(p!=500):
		index = random.randint(1, len(new_data)-1)
		if(new_data[index][4]=='1'):
			new_data = np.delete(new_data, (index), axis=0)
			#new_data.pop(index)
			p +=1
	return new_data


def main():
	global a
	global b
	data = []
	with open('data-2_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for row in csv_reader:
			data.append(row)
	fields = data[0]
	data = np.array(data[1:], dtype=object)
	print(data.shape, fields)

	#data = random_undersampler(data)
	


	#words = filterData(data)

	#******************#
	words1 = filterData(data[:int(0.8*len(data))])
	#print(words.shape)

	print('$$$$$$$$$')
	print(len(a))
	print(len(b))
	print('$$$$$$$$$')

	c = list(b)

	final_x_train = []
	final_y_train = []

	for i in range(len(words1)):
		final_x_train.append(encode_onehot(words1[i][1], c))
		final_y_train.append(words1[i][4])

	a= []
	b = []
	b = set(b)

	ones = final_y_train.count('1')
	zeros = final_y_train.count('0')
	negs = final_y_train.count('-1')
		
	words2 = filterData(data[int(0.8*len(data)):])

	print('2$$$$$$$$$')
	print(len(a))
	print(len(b))
	print(len(c))
	print('2$$$$$$$$$')



	final_x_test =[]
	final_y_test =[]
	for i in range(len(words2)):
		final_x_test.append(encode_onehot(words2[i][1], c))
		final_y_test.append(words2[i][4])


	ones += final_y_test.count('1')
	zeros += final_y_test.count('0')
	negs += final_y_test.count('-1')


	# final_x_test = []
	# final_y_test = []
	# for i in range(int(0.75*len(data)), len(data)):
	# 	final_x_test.append(encode_onehot(words[i][1], c[:int(0.5*len(c))]))
	# 	final_y_test.append(data[i][4])

	print('##FINAL DATA##')
	print(final_x_train[0].shape)
	print(final_x_test[0].shape)
	print('####')

	# words = filterData(data)

	# x_train = []
	# y_train = []
	# for i in range(len(data)):
	# 	x_train.append(words[i][1])
	# 	y_train.append(data[i][4])

	# ones = y_train.count('1')
	# zeros = y_train.count('0')
	# negs = y_train.count('-1')


	# x_train = np.array(x_train)
	# y_train = np.array(y_train)

	# # 10-Fold Cross Validation
	# kf = KFold(n_splits=10)
	# kf.get_n_splits(x_train)

	# precision_nb = np.array([0.0,0.0])
	# recall_nb = np.array([0.0,0.0]) 
	# f_score_nb = np.array([0.0,0.0])
	# precision_svm = np.array([0.0,0.0])
	# recall_svm = np.array([0.0,0.0]) 
	# f_score_svm = np.array([0.0,0.0])
	# count = 1
	# for train_index, test_index in kf.split(x_train):
	# 	x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
	# 	y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

	# 	onehot_enc = MultiLabelBinarizer()
	# 	onehot_enc.fit(x_train)

	# 	bnbc = BernoulliNB(binarize=None)
	# 	bnbc.fit(onehot_enc.transform(x_train_kf), y_train_kf)

	# 	predicted_y = bnbc.predict(onehot_enc.transform(x_test_kf))
		
	# 	print(onehot_enc.transform(x_test_kf))
	# 	print(onehot_enc.transform(x_test_kf).shape)
	# 	print ('length of predicted',len(predicted_y))


	# 	score = bnbc.score(onehot_enc.transform(x_test_kf), y_test_kf)

	# 	precision_nb += precision_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
	# 	recall_nb += recall_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
	# 	f_score_nb += f1_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)

	# 	print(count, "Naive Bayesian Accuracy: ", score)
	# 	# print(bnbc.predict(onehot_enc.transform(x_test_kf)))

	# 	lsvm = LinearSVC()
	# 	lsvm.fit(onehot_enc.transform(x_train_kf), y_train_kf)

	# 	predicted_y = lsvm.predict(onehot_enc.transform(x_test_kf))
	# 	precision_svm += precision_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
	# 	recall_svm += recall_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
	# 	f_score_svm += f1_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)

	# 	score = lsvm.score(onehot_enc.transform(x_test_kf), y_test_kf)
	# 	print(count, "Linear SVM Accuracy: ", score)
	# 	print("")
	# 	count += 1

	# print('NB Avg. Precisions', precision_nb/10)

	# print('NB Avg. Recalls', recall_nb/10)

	# print('NB Avg. F-Scores', f_score_nb/10)

	# print('SVM Avg. Precisions', precision_svm/10)
	# print('SVM Avg. Recalls', recall_svm/10)
	# print('SVM Avg. F-Scores', f_score_svm/10)


	# Neural network
	batch_size = 300
	tf.reset_default_graph()

	vocab_len = len(final_x_train[0])
	inputs_ = tf.placeholder(dtype=tf.float32, shape=[None, vocab_len], name="inputs")
	targets_ = tf.placeholder(dtype=tf.float32, shape=[None, 3], name="targets")

	h1 = tf.layers.dense(inputs_, 500, activation=tf.nn.relu)
	logits = tf.layers.dense(h1, 3, activation=None)
	output = tf.nn.sigmoid(logits)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets_))

	optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(targets_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())

	for l in range(10):
		#x_train_kf, x_test_kf = final_x_train[train_index], final_x_train[test_index]
		#y_train_kf, y_test_kf = final_y_train[train_index], final_x_train[test_index]
		#number of epochs
		for i in range(20):

			#Check strt
			#xtrn, xtst, ytrn, ytst = train_test_split(x_train, y_train, test_size=0.2)

			##Check end
			for x_batch, y_batch in get_batch(final_x_train, label2bool(final_y_train), batch_size):         
				loss_value, _ = sess.run([loss, optimizer], feed_dict={
					inputs_: x_batch,
					targets_: y_batch
				})

		test_acc = sess.run(accuracy, feed_dict={
			inputs_: final_x_test,
			targets_: label2bool(final_y_test)
		})

		print("Test Accuracy: {}".format(test_acc))

		train_acc = sess.run(accuracy, feed_dict={
			inputs_: final_x_train,
			targets_: label2bool(final_y_train)
		})

		print("Train Accuracy: {}".format(train_acc))

	print('Ones', ones)
	print('Zeros', zeros)
	print('negatives', negs)
	# print(accuracy)

if __name__  == "__main__":
	main()




