import numpy
import pickle
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain3.structure import LinearLayer

# region Constants
HISTORY_FILENAME = 'history.txt'
NETDUMP_FILENAME = 'networkdump.txt'
# endregion

# region Populate input file
# fp = open('history.txt', 'w')
# for x in range(100):
#     fp.write("%d " % (22 + random.randrange(-1, 2, 1)))
# fp.close()
# endregion

# region Initialize
inp = numpy.loadtxt(HISTORY_FILENAME, int)

inputLength = len(inp)
observeLength = 10
predictionLength = 1

trainDataSet = SupervisedDataSet(observeLength, predictionLength)

for n in range(inputLength):
    if n + (observeLength - 1) + predictionLength < inputLength:
        trainDataSet.addSample(inp[n:n + observeLength], inp[n + 1:n + 1 + predictionLength])

net = buildNetwork(observeLength, 20, predictionLength, outclass=LinearLayer, bias=True, recurrent=True)
trainer = BackpropTrainer(net, trainDataSet)
trainer.trainEpochs(100)
# endregion

# region Test
ts = UnsupervisedDataSet(observeLength, )
ts.addSample(inp[-observeLength:])

print([int(round(i)) for i in net.activateOnDataset(ts)[0]])
# endregion

# region Save the network
dmpFile = open(NETDUMP_FILENAME, 'w')
pickle.dump(net, dmpFile)
dmpFile.close()
# endregion
