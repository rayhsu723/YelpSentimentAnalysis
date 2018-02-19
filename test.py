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



# takes in review.json and cleans it up because it isn't in correct json format
# --- PARAMETERS ---
# filename:       should ne 'review.json', but it can technically take any file
# outfile:        the file that we output the correct JSON
# num_of_reviews: determines how many reviews we want to get from review.json
def clean_data(filename, outfile, num_of_reviews):
	out = open(outfile, 'w')
	print('[', file=output)
	file = open(filename, 'r')

	for r in range(num_of_reviews-1):
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
	data = []
	words = Counter()
	wordlist = []
	bow = []

	def initialize(self, json_file):
		with open(json_file, 'r') as f:
			self.data = json.load(f)
			self.data = [x for x in self.data if x['stars']!=3]
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

			# sent = element['text']
			# sent = sent.translate(str.maketrans('','',string.punctuation))
			# sent = sent.translate(str.maketrans('','','1234567890'))




	def tokenize(self, tokenizer=nltk.word_tokenize):
		stopwords = nltk.corpus.stopwords.words('english')
		whitelist = ["n't", "not"]
		stopwords = [word for word in stopwords if word not in whitelist]


		# for row in self.data:
		# 	sent = row['text'].lower()
		# 	sent = sent.translate(str.maketrans('','',string.punctuation))
		# 	sent = sent.translate(str.maketrans('','','1234567890'))
		# 	row['text'] = tokenizer(sent)
		# 	row['text'] = [word for word in row['text'] if word not in stopwords]

	def build_word_list(self, min_occurences=0, max_occurences=1000):
		for row in self.data:
			self.words.update(row['text'])

		self.wordlist = [k for k,v in self.words.most_common() if min_occurences < v < max_occurences]
		self.wordlist = sorted(self.wordlist)
		# stopwords = nltk.corpus.stopwords.words('english')
		# whitelist = ["n't", "not"]
		# for idx, stop_word in enumerate(stopwords):
		# 	if stop_word not in whitelist:
		# 		del self.words[stop_word]
 

	def build_bow(self):
		# for data in self.data:
		# 	temp = []
		# 	for word in wordlist:
		# 		if word in data['text']:
		# 			temp.append(1)
		# 		else:
		# 			temp.append(0)
		# 	temp.append(data['stars'])
		# 	self.bow.append(temp)
		# # self.words = [Counter(x['text']) for x in self.data]
		# # sumbags = sum(self.words, Counter())
		# # print(sumbags)
		text = [x['text'] for x in self.data]
		vectorizer = CountVectorizer(analyzer='word')
		X = vectorizer.fit_transform(text)
		# print(X.toarray())
		# print(vectorizer.get_feature_names())
		Y = [x['stars'] for x in self.data]

		return (X,Y)


if __name__ == '__main__':
	
	


	stuff = YelpData_Init()
	stuff.initialize('r.json')
	stuff.tokenize()
	stuff.build_word_list()
	# print(len(stuff.words))
	# print(stuff.words.most_common(5))
	stuff.build_bow()
















# file_name = "review.json"
# changed_file = 'r.json'



# output = open('r.json', 'w')
# print('[', file=output)

# file = open(file_name, 'r')

# for x in range(999):
# 	line = file.readline()
# 	line = line.rstrip() + ','
# 	print(line, file=output)
# line = file.readline()
# line = line.rstrip()
# print(line, file=output)
# print(']',file=output)

# file.close()
# output.close()

# datastore = ''

# with open(changed_file, 'r') as f:
# 	datastore = json.load(f)
# 	datastore = [x for x in datastore if x['stars']!=3]
# 	for element in datastore:
# 		del element['review_id']
# 		del element['user_id']
# 		del element['business_id']
# 		del element['date']
# 		if element['stars'] == 4 or element['stars'] == 5:
# 			element['stars'] = "p"
# 		elif element['stars'] == 2 or element['stars'] == 1:
# 			element['stars'] = "n"

# for stuff in datastore:
# 	if stuff['stars'] == 3:
# 		print(stuff['stars'])
# 	else:
# 		print(stuff['stars'])




