import gensim
from pybrain.tools.customxml  import NetworkReader
import numpy as np
#load the gensim model
model = gensim.models.Word2Vec.load('news.en.model')

fnn = NetworkReader.readFrom('oliv.xml') 
word =np.array([model['us-led_JJ'],model['american-led_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['white_JJ'],model['black_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['local_JJ'],model['international_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['bike_NN'],model['motorcycle_NN']] )
pred= fnn.activate(word.flatten())
print pred
# and 
#bike and motorcycle 
#bike and banana_NN
word =np.array([model['polite_JJ'],model['impolite_JJ']] )
pred= fnn.activate(word.flatten())
print pred



word =np.array([model['intelligent_JJ'],model['stupid_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['smart_JJ'],model['intelligent_JJ']] )
pred= fnn.activate(word.flatten())
print pred

word =np.array([model['smart_JJ'],model['stupid_JJ']] )
pred= fnn.activate(word.flatten())
print pred