from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer,SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.tools.customxml  import NetworkReader
from pybrain.tools.customxml  import NetworkWriter

import numpy as np


# ds = SupervisedDataSet(2, 1)

# input =np.array([[0],[0]] )
# ds.addSample((input.flatten()), (0,))
# ds.addSample((0, 1), (1,))
# ds.addSample((1, 0), (1,))
# ds.addSample((1, 1), (0,))

# Produce two new datasets, the first one containing the fraction given by proportion of the samples.
# splitWithProportion(proportion=0.5)
# print len(ds)

# tstdata, trndata = alldata.splitWithProportion( 0.25 )

# for input, target in ds:
	# print input,target
	
# print ds['input']

# print ds['target']

#hidden class by default sigmoid
all_data=ClassificationDataSet.loadFromFile("nn-data")

# tstdata, trndata = all_data.splitWithProportion( 0.25 )
tstdata_temp, partdata_temp = all_data.splitWithProportion( 0.25 )

trndata_temp,validata_temp = partdata_temp.splitWithProportion(0.50)

tstdata = ClassificationDataSet(200, 1, nb_classes=2)
for n in xrange(0, tstdata_temp.getLength()):
	tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(200, 1, nb_classes=2)
for n in xrange(0, trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
	
validata= ClassificationDataSet(200, 1, nb_classes=2)
for n in xrange(0, validata_temp.getLength()):
    trndata.addSample( validata_temp.getSample(n)[0], validata_temp.getSample(n)[1] )
	



trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()

# for input, target in trndata:
	# print len(input),target

# split up training data for cross validation
print "Split data into training and test sets..."

net = buildNetwork(200, 134, 2, bias=True, outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, dataset=trndata)
print "training for {} epochs..."

trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )



trnresult = percentError( trainer.testOnClassData(),trndata['class'] )
tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,           "  test error: %5.2f%%" % tstresult
NetworkWriter.writeToFile(net, 'oliv-x2-80.xml')


# predict using test data
# print "Making predictions..."
# ypreds = []
# ytrues = []
# for i in range(Xtest.getLength()]):
    # pred = fnn.activate(getSample(i)[0])
    # ypreds.append(pred.argmax())
    # ytrues.append(ytest[i])
# print "Accuracy on test set: %7.4f" % accuracy_score(ytrues, ypreds, 
                                                     # normalize=True)


	
