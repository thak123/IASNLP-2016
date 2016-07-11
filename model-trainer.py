import logging
import gensim


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec .LineSentence('corpus-processed-1M.txt')
model = gensim.models.word2vec .Word2Vec(sentences, sg=1,size=25, window=8)
model.save('news.en.model.25')

