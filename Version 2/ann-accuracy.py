import gensim
from pybrain.tools.customxml  import NetworkReader
import numpy as np
#load the gensim model
model = gensim.models.Word2Vec.load('news.en.model')

fnn = NetworkReader.readFrom('clean-oliv.xml') 
# word =np.array([model['us-led_JJ'],model['american-led_JJ']] )
# pred= fnn.activate(word.flatten())
# print pred

# word =np.array([model['white_JJ'],model['black_JJ']] )
# pred= fnn.activate(word.flatten())
# print pred

#local and international
#bike and motorcycle 
#bike and banana_NN
# word =np.array([model['polite_JJ'],model['impolite_JJ']] )
# pred= fnn.activate(word.flatten())
# print pred 100-1 010-2 100-4

word =np.array([model['fresh_JJ'],model['preserved_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['cheap_JJ'],model['expensive_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['bad_JJ'],model['terrible_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['liberated_JJ'],model['multiparty_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['exponential_JJ'],model['precipitous_JJ']] )
pred= fnn.activate(word.flatten())
print pred
	
