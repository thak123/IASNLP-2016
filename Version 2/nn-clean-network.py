from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer 
from pybrain.datasets import ClassificationDataSet

from pybrain.utilities import percentError
from pybrain.tools.customxml  import NetworkWriter
from pybrain.tools.customxml  import NetworkReader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import gensim 
import codecs
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load the gensim model
model = gensim.models.Word2Vec.load('news.en.model')

#read csv file
df = pd.read_csv('dataset-3-class.txt',header=None,delimiter=r"\s+")

#define the dimensions for input output
inputdim=200
outputdim=3
noofhiddenunits=136 #136 #25 #134 #140 #67
ds = ClassificationDataSet(inputdim,1,nb_classes=3)

#create the classification dataset
for index, row in df.iterrows():
	word =np.array([model[row[0]],model[row[1]]] )
	label =0 
	if row[2]=="SYN":
		label =0 
	elif row[2]=="ANT":
		label =1
	elif row[2]=="ELS":
		label =2
	ds.addSample((word.flatten()), (label,))

# datas= copy.deepcopy(ds)
#convert the one dimensions of output to multidimesions 0 -> 100, 1->010
# ds._convertToOneOfMany()

print ds
# for i in range(len(ds['target'])):
	# print ds.getSample(i)[1] ,datas.getSample(i)[1]
	
print len(ds)	
# split up training data for cross validation
print "Split data into training and test sets..."

#split data into train and validate and test 60/40(50:50)
traindata_temp,partdata= ds.splitWithProportion(0.60)

testdata_temp,validata_temp= partdata.splitWithProportion(0.50)

print len(traindata_temp)
print len(testdata_temp)
print len(validata_temp)


testdata = ClassificationDataSet(inputdim, 1, nb_classes=outputdim)
for n in xrange(0, testdata_temp.getLength()):
	testdata.addSample( testdata_temp.getSample(n)[0], testdata_temp.getSample(n)[1] )

traindata = ClassificationDataSet(inputdim, 1, nb_classes=outputdim)
for n in xrange(0, traindata_temp.getLength()):
    traindata.addSample( traindata_temp.getSample(n)[0], traindata_temp.getSample(n)[1] )
	
validata= ClassificationDataSet(inputdim, 1, nb_classes=outputdim)
for n in xrange(0, validata_temp.getLength()):
    traindata.addSample( validata_temp.getSample(n)[0], validata_temp.getSample(n)[1] )
	

#convert the one dimensions of output to multidimesions 0 -> 100, 1->010
traindata._convertToOneOfMany()
testdata._convertToOneOfMany()
validata._convertToOneOfMany()

#build a network
net = buildNetwork(inputdim, noofhiddenunits, outputdim, bias=True, outclass=SoftmaxLayer)
#defing a backpropagator
trainer = BackpropTrainer(net, dataset=traindata)
print "training for {} epochs..."

trainerror,valerror=trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )

# plt.plot(trainerror,'b',valerror,'r')
# plt.show()

# trainer.trainOnDataset(traindata,500)



trnresult = percentError( trainer.testOnClassData(),traindata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=testdata ), testdata['class'] )
# valiresult = percentError( trainer.testOnClassData(dataset=validata ), validata['class'] )
 
print "epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,           "  test error: %5.2f%%" % tstresult
# ,    "valid error:%5.2ff%" %valiresult


# if  os.path.isfile('clean-oliv.xml'): 
 # fnn = NetworkReader.readFrom('clean-oliv.xml') 
# else:
 # fnn = buildNetwork( traindata.indim, 64 , traindata.outdim, outclass=SoftmaxLayer )
 # NetworkWriter.writeToFile(net, 'clean-oliv.xml')
NetworkWriter.writeToFile(net, 'clean-oliv-136.xml')