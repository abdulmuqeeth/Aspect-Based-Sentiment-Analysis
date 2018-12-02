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
	for i in range(len(data)):
		x_train.append(words[i][1])
		y_train.append(data[i][4])
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	# 10-Fold Cross Validation
	kf = KFold(n_splits=10)
	kf.get_n_splits(x_train)

	precision_nb = np.array([0.0,0.0])
	recall_nb = np.array([0.0,0.0]) 
	f_score_nb = np.array([0.0,0.0])
	precision_svm = np.array([0.0,0.0])
	recall_svm = np.array([0.0,0.0]) 
	f_score_svm = np.array([0.0,0.0])
	count = 1
	for train_index, test_index in kf.split(x_train):
		x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
		y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

		onehot_enc = MultiLabelBinarizer()
		onehot_enc.fit(x_train)

		bnbc = BernoulliNB(binarize=None)
		bnbc.fit(onehot_enc.transform(x_train_kf), y_train_kf)

		predicted_y = bnbc.predict(onehot_enc.transform(x_test_kf))
		print(onehot_enc.transform(x_test_kf))
		print(onehot_enc.transform(x_test_kf).shape)
		print ('length of predicted',len(predicted_y))
		score = bnbc.score(onehot_enc.transform(x_test_kf), y_test_kf)

		precision_nb += precision_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
		recall_nb += recall_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
		f_score_nb += f1_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)

		print(count, "Naive Bayesian Accuracy: ", score)
		# print(bnbc.predict(onehot_enc.transform(x_test_kf)))

		lsvm = LinearSVC()
		lsvm.fit(onehot_enc.transform(x_train_kf), y_train_kf)

		predicted_y = lsvm.predict(onehot_enc.transform(x_test_kf))
		precision_svm += precision_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
		recall_svm += recall_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)
		f_score_svm += f1_score(y_test_kf, predicted_y, labels=['-1','1'],average=None)

		score = lsvm.score(onehot_enc.transform(x_test_kf), y_test_kf)
		print(count, "Linear SVM Accuracy: ", score)
		print("")
		count += 1

	print('NB Avg. Precisions', precision_nb/10)

	print('NB Avg. Recalls', recall_nb/10)

	print('NB Avg. F-Scores', f_score_nb/10)

	print('SVM Avg. Precisions', precision_svm/10)
	print('SVM Avg. Recalls', recall_svm/10)
	print('SVM Avg. F-Scores', f_score_svm/10)


	# Neural network
	batch_size = 300
	tf.reset_default_graph()

	vocab_len = len(onehot_enc.classes_)
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

	for train_index, test_index in kf.split(x_train):
		x_train_kf, x_test_kf = x_train[train_index], x_train[test_index]
		y_train_kf, y_test_kf = y_train[train_index], y_train[test_index]

		for i in range(10):
			for x_batch, y_batch in get_batch(onehot_enc.transform(x_train_kf), label2bool(y_train_kf), batch_size):         
				loss_value, _ = sess.run([loss, optimizer], feed_dict={
					inputs_: x_batch,
					targets_: y_batch
				})

		test_acc = sess.run(accuracy, feed_dict={
			inputs_: onehot_enc.transform(x_test_kf),
			targets_: label2bool(y_test_kf)
		})

		print("Test Accuracy: {}".format(test_acc))


	# print(accuracy)

if __name__  == "__main__":
	main()




