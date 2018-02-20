import json
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
# from pprint import pprint
import os
import re
import numpy as np
import string
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import nltk.classify.util


# takes in review.json and cleans it up because it isn't in correct json format
# --- PARAMETERS ---
# filename:       should ne 'review.json', but it can technically take any file
# outfile:        the file that we output the correct JSON
# num_of_reviews: determines how many reviews we want to get from review.json
def clean_data(filename, outfile, num_of_reviews=10000):
	out = open(outfile, 'w')
	print('[', file=output)
	file = open(filename, 'r')

	for r in range(num_of_reviews - 1):
		line = file.readline()
		line = line.rstrip() + ','
		print(line, file=out)
	line = file.readline()
	line = line.rstrip()
	print(line, file=out)
	print(']', file=out)
	file.close()
	out.close()


class YelpData_Init():
	def __init__(self):
		# contains a list of dictionaries. dictionaries have two keys (stars: positive or negative, text: self-explanatory)
		self.data = []
		# a total count of all the words in every review
		self.words = Counter()
		# every single word that appears in the reviews
		self.wordlist = []

	# file is too large and in incorrect format so we need to clean it up
	def initialize(self, json_file):
		with open(json_file, 'r', encoding='utf8') as f:
			self.data = json.load(f)
			self.data = [x for x in self.data if x['stars'] != 3]
		for element in self.data:
			element.pop('review_id', None)
			element.pop('user_id', None)
			element.pop('business_id', None)
			element.pop('date', None)
			element.pop('funny', None)
			element.pop('cool', None)
			element.pop('useful', None)
			if element['stars'] == 4 or element['stars'] == 5:
				element['stars'] = 'p'
			elif element['stars'] == 2 or element['stars'] == 1:
				element['stars'] = 'n'


	# for now, this just removes stopwords. Used to actually tokenize, but the count vectorizer in build_bow already does that for us
	def tokenize(self, tokenizer=nltk.word_tokenize):
		stopwords = nltk.corpus.stopwords.words('english')
		whitelist = ["n't", "not"]
		stopwords = [word for word in stopwords if word not in whitelist]


	# builds the word list
	def build_word_list(self, min_occurences=0, max_occurences=1000):
		for row in self.data:
			self.words.update(row['text'])

		self.wordlist = [k for k, v in self.words.most_common() if min_occurences < v < max_occurences]
		self.wordlist = sorted(self.wordlist)


	# uses a count vectorizer to create a sparse matrix of reviews
	def build_bow(self):
		text = [x['text'] for x in self.data]
		vectorizer = CountVectorizer(analyzer='word')
		X = vectorizer.fit_transform(text)
		# print(X.toarray())
		# print(vectorizer.get_feature_names())
		Y = [x['stars'] for x in self.data]

		return (X, Y)

	# give it a classifier from sklearn that will be used to determine accuracy, recall, precision, and f1 score
	# accuracy: self-explanatory
	# recall: how many relevant items are selected?
	# precision: how many selected items are relevant?
	# f1 score: harmonic mean of precision and recall. Also kinda like an accuracy rating
	def classify(self, classifier):
		classifier_name = str(type(classifier).__name__)
		print("CLASSIFIER = " + classifier_name)

		X, Y = self.build_bow()
		Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.33, random_state=42)

		model = classifier.fit(Xtr, Ytr)
		predicted = model.predict(Xte)

		list_of_labels = sorted(list(set(Ytr)))

		accuracy = accuracy_score(Yte, predicted)
		recall = recall_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)
		precision = precision_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)
		f1 = f1_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)

		print("=================== Results ===================")
		print("           Negative    Positive")
		print("F1       " + str(f1))
		print("Precision" + str(precision))
		print("Recall   " + str(recall))
		print("Accuracy " + str(accuracy))
		print("===============================================")



if __name__ == '__main__':
	stuff = YelpData_Init()
	stuff.initialize('set10k.json')
	stuff.tokenize()
	stuff.build_word_list()
	# print(len(stuff.words))
	# print(stuff.words.most_common(5))
	stuff.classify(MultinomialNB())
	stuff.classify(LogisticRegression())
