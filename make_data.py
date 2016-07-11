# a program to prepare the training data for synonyms and other relations

import nltk
from nltk.corpus import wordnet as wn 
from collections import defaultdict

# function to read from text files and returns a list of lines 
def text_to_lines( file_name ):
	with open( file_name, 'r') as txt_file:
		text = txt_file.read()

	return text.split('\n')[:-1]

# function to return intersection of two lists 
def intersect( list_1, list_2 ):
	new_list = []

	# convert words in list_2 (output of wordnet.synsets) from unicode to string
	list_2_str = []

	for word in list_2:
		list_2_str.append( str(word) )

	for e in list_1:
		if e in list_2_str:
			new_list.append(e)

	return new_list

# a function to check whether or not a word exist in our corpus 
def isInCorpus( word ):
	if word in word_clusters.keys():
		return True
	else:
		return False

# read the word clusters from disk (word vectors from word2vec)
corpus = text_to_lines('sim-scores-adj-200.txt')

word_clusters = {}

for line in corpus:
	words = line.split()

	if words:
		word_clusters[words[0]] = words[1:]

# get adjectives from SimLex-999 lexicon and find its synonyms from wordnet
simlex = text_to_lines('simlex-adj.txt')

simlex_data= []

for instance in simlex:
	word_1, word_2, label = instance.split()
	if isInCorpus(word_1) and isInCorpus(word_2) :
		w1, w2 = sorted([word_1, word_2])
		#print word_1, word_2, w1, w2
		simlex_data.append([w1, w2, label])

# dict to store synonyms dict[word] = [list of synonyms]
word_syns = defaultdict(list)

# dict to store antonyms dict[word] = [list of antonyms]
word_ants = defaultdict(list)

# for syn in wn.synsets(s[0], pos=wn.ADJ):
# 		for l in syn.lemmas():
# 			if l.antonyms():
# 			    antonyms.append(l.antonyms()[0].name())
# 	for ant in (set(antonyms)):
# 		instances.append( (s[0], ant, 'ANT') )

for instance in simlex_data:
	# for each word pair get synonyms from wordnet
	word = instance[0]
	syns = wn.synsets(word[:word.find('_')], pos=wn.ADJ)

	# if syns is not an empty list, iterate 
	if syns:
		for s in syns:
			for l in s.lemmas():
				word_syns[ word ].append( l.name().lower() + '_JJ' )

				if l.antonyms():
					word_ants[ word ].append( l.antonyms()[0].name().lower() + '_JJ' )


	word = instance[1]
	syns = wn.synsets(word[:word.find('_')], pos=wn.ADJ)

	# if syns is not an empty list, iterate 
	if syns:
		for s in syns:
			for l in s.lemmas():
				word_syns[ word ].append( l.name().lower() + '_JJ' )

				if l.antonyms():
					word_ants[ word ].append( l.antonyms()[0].name().lower() + '_JJ' )
#print word_syns

final_syns = defaultdict(list)
final_ants = defaultdict(list)

for word in word_syns.keys():
	final_syns[word].extend(intersect(word_clusters[word], word_syns[word]))

for word in word_ants.keys():
	final_ants[word].extend(intersect(word_clusters[word], word_ants[word]))

counter = 0

data = []
for word, synonyms in final_syns.items():
	if len(synonyms)>0:
		counter += 1
		for s in synonyms:
			w1, w2 = sorted([word, s])
			data.append([w1, w2,'SYN'])

for word, antonyms in final_ants.items():
	if len(antonyms)>0:
		counter += 1
		for s in antonyms:
			w1, w2 = sorted([word, s])
			data.append([w1, w2,'ANT'])

final_list = []

for d in data:
	if d not in final_list:
		final_list.append(d)

count_syn = 0
count_ant = 0

for point in final_list:
	print point[0] + '\t' + point[1] + '\t' + point[2]
	if point[2] == 'SYN':
		count_syn += 1
	else:
		count_ant += 1

# print len(data)
# print len(final_list)
# print count_syn
# print count_ant

