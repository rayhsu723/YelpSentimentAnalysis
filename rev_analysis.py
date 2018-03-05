import json
import ijson
import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
plt.interactive(False)

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

# from pprint import pprint
import os
import re
import numpy as np
import string
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from sklearn.feature_extraction.text import TfidfTransformer

import nltk.classify.util
from nltk.sentiment.util import mark_negation


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# takes in review.json and cleans it up because it isn't in correct json format
# --- PARAMETERS ---
# filename:       should ne 'review.json', but it can technically take any file
# outfile:        the file that we output the correct JSON
# num_of_reviews: determines how many reviews we want to get from review.json
def clean_data(filename, outfile, num_of_reviews=10000):
	out = open(outfile, 'w')
	print('[', file=out)
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
		self.stars = {1: 0, 2:0, 3:0, 4:0, 5:0}
		self.seed=42
		self.X = None
		self.Y = None

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
			self.stars[element['stars']] += 1
			if element['stars'] == 4 or element['stars'] == 5:
				element['stars'] = 'p'
			elif element['stars'] == 2 or element['stars'] == 1:
				element['stars'] = 'n'


	# for now, this just removes stopwords. Used to actually tokenize, but the count vectorizer in build_bow already does that for us
	def tokenize(self, tokenizer=nltk.word_tokenize):
		stopwords = nltk.corpus.stopwords.words('english')
		whitelist = ["n't", "not"]
		stopwords = [word for word in stopwords if word not in whitelist]

		# for x in self.data:
		# 	temp = tokenizer(x['text'])
		# 	for word in list(temp):
		# 		if word in stopwords:
		# 			temp.remove(word)
		# 	x['text'] = ' '.join(temp)
		for x in self.data:
			temp = tokenizer(x['text'])
			for word in list(temp):
				if word.lower() in stopwords or word.lower() not in self.wordlist:
					temp.remove(word)
			x['text'] = ' '.join(temp)
 

	# builds the word list
	def build_word_list(self, min_occurences=0, max_occurences=100000):
		for row in self.data:
			self.words.update(nltk.word_tokenize(row['text'].lower()))
		self.wordlist = [k for k, v in self.words.most_common() if min_occurences < v < max_occurences]
		self.wordlist = sorted(self.wordlist)
		
		# stopwords = nltk.corpus.stopwords.words('english')
		# whitelist = ["n't", "not"]

		# for idx, stop_word in enumerate(stopwords):
		# 	if stop_word not in whitelist:
		# 		del self.words[stop_word]

		# for punc in string.punctuation:
		# 	if punc in self.words:
		# 		del self.words[punc]
		print(len(self.wordlist))


	# uses a count vectorizer to create a sparse matrix of reviews
	def build_bow(self, POS=True, negation=True, ngram_range=(1,2), universal=False):
		text = []
		# print(self.data[0]['text'])

		if negation==True:
			text = [' '.join(mark_negation(x['text'].replace('.', ' .').split())) for x in self.data]
		else:
			text = [x['text'] for x in self.data]

		# print(len(text))

				#Tagging for part of speech. Has universal flag included

		if POS==True:
			text = []
			if universal:
				pos = [pos_tag(word_tokenize(x['text']), tagset='universal') for x in self.data]

			else:
				pos = [pos_tag(word_tokenize(x['text'])) for x in self.data]

			for pair in pos:
				text.append(' '.join(['_'.join(list(p)) for p in pair]))

		# mark_negation(text, shallow=True)

		vectorizer = CountVectorizer(ngram_range=ngram_range, token_pattern=r'\b\w+\b', min_df=1)

		X = vectorizer.fit_transform(text)
		# print(vectorizer.get_feature_names())
		Y = [x['stars'] for x in self.data]


		return (X, np.asarray(Y))

	#
	def pos_tagging(self, universal = False):
		text = []
		if universal:
			pos = [pos_tag(word_tokenize(x['text']), tagset='universal') for x in self.data]

		else:
			pos = [pos_tag(word_tokenize(x['text'])) for x in self.data[:2]]

		for pair in pos:
			text.append(' '.join(['_'.join(list(p)) for p in pair]))




	# give it a classifier from sklearn that will be used to determine accuracy, recall, precision, and f1 score
	# accuracy: self-explanatory
	# recall: how many relevant items are selected?
	# precision: how many selected items are relevant?
	# f1 score: harmonic mean of precision and recall. Also kinda like an accuracy rating
	def classify(self, classifiers, POS=True, negation=True, ngram_range=(1,2), min_df=1, max_df=100000):
		self.initialize('set50k.json')
		self.build_word_list(min_occurences=min_df, max_occurences=max_df)
		self.tokenize()
		# self.build_word_list(min_occurences=min_df, max_occurences=max_df)
		self.X, self.Y = self.build_bow(POS=True, negation=True, ngram_range=ngram_range)
		Xtr, Xte, Ytr, Yte = train_test_split(self.X, self.Y, test_size=.33, random_state=self.seed)
		# print(Xtr.shape,Xte.shape)
		for classifier in classifiers:
			classifier_name = str(type(classifier).__name__)  
			print("CLASSIFIER = " + classifier_name)

			model = classifier.fit(Xtr, Ytr)
			predicted = model.predict(Xte)

			list_of_labels = sorted(list(set(Ytr)))

			# accuracy = accuracy_score(Yte, predicted)
			recall = recall_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)
			precision = precision_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)
			f1 = f1_score(Yte, predicted, pos_label=None, average=None, labels=list_of_labels)

			folds = 10
			scores = cross_val_score(model, self.X, self.Y, cv = folds)

			print("=================== Results ===================")
			print("           Negative    Positive")
			print("F1       " + str(f1))
			print("Precision" + str(precision))
			print("Recall   " + str(recall))
			# print("Accuracy " + str(accuracy))
			print("Cross-validation with {} folds:".format(folds))
			print("\t Scores: {}".format(scores))
			print("\t Accuracy: {} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
			print("===============================================")

	def dataStats(self):
		print(self.stars)
		reviews, vocabulary = self.X.shape
		print(self.X.shape)
		count = 0
		total = 0
		for review in self.data:
			total += len(review['text'].split())
			count += 1

		avg_length = total/count
		print(avg_length)

		plt.bar(list(self.stars.keys()),self.stars.values(), color = 'g', width = .5)
		plt.title("Histogram of Rated Stars")
		plt.xlabel("Stars")
		plt.ylabel("Number of Reviews")
		plt.show()


if __name__ == '__main__':

	# clean_data('review.json', 'set30k.json', 30000)
	sentiment = YelpData_Init()


	# sentiment.initialize('set1000.json')
	# sentiment.build_word_list(min_occurences=3)

	# sentiment.tokenize()

	sentiment.classify([LogisticRegression()], min_df=10)

	#sentiment.dataStats()

#sentiment.classify([LogisticRegression(), MultinomialNB(), RandomForestClassifier(n_estimators=25,max_depth=75,max_features=.75), AdaBoostClassifier()], min_df=5)


