from nltk.corpus import wordnet as wn

word_list=[]
print '1'
words = wn.synsets('wonderful',pos=wn.ADJ)

for word in words:
	print word.lemmas()[0].antonyms()
	word_list.extend(word.lemma_names())


print word_list