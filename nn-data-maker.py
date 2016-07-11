# -*- coding: utf-8 -*-
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet

import gensim
import codecs
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np
import pandas as pd

#load the gensim model
model = gensim.models.Word2Vec.load('news.en.model')

#code to read csv file
df = pd.read_csv('traing-data-jj.txt',header=None,delimiter=r"\s+")

#hardcoded value ????
ds = ClassificationDataSet(200,1,nb_classes=2)


for index, row in df.iterrows():
	word =np.array([model[row[0]],model[row[1]]] )
	label =0 if row[2]=="SYN" else 1
	ds.addSample((word.flatten()), (label,))

for input, target in ds:
	print target

ds._convertToOneOfMany()	

# for input, target in ds:
	# your_word_vector = np.array([input[:100]], dtype='f')
	# print target
	# print len(input),target,input[:100],model.most_similar(positive=[your_word_vector], topn=1)
		 

for input, target in ds:
	print target
	
# ds.saveToFile('nn-data')