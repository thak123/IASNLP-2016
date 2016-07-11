# -*- coding: utf-8 -*-


import gensim
import codecs
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


model = gensim.models.Word2Vec.load('news.en.model')
vocab = list(model.vocab.keys())

with codecs.open( "word-similarity.txt",'w') as f:
    for words in vocab:
        vectors = model.most_similar(words)
        f.write(words.encode('utf-8'))
        f.write('\n')
        for vector in vectors:
            f.write('\t')
            f.write(vector[0].encode('utf-8'))
            f.write('\n')
        f.write('\n')




