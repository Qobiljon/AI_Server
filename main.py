import numpy
import pickle
from os.path import isfile
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain3.structure import LinearLayer

# region Constants
HISTORY_FILENAME = 'history.txt'
NETDUMP_FILENAME = 'networkdump.txt'

OBSERVE_LENGTH = 10
PREDICT_LENGTH = 1
# endregion

# region Populate input file
# fp = open('history.txt', 'w')
# for x in range(100):
#     fp.write("%d " % (22 + random.randrange(-1, 2, 1)))
# fp.close()
# endregion

# region Initialize
net = None
if isfile(NETDUMP_FILENAME):
    netDumpFile = open(NETDUMP_FILENAME, 'rb')
    net = pickle.load(netDumpFile)
    netDumpFile.close()
else:
    inp = numpy.loadtxt(HISTORY_FILENAME, int)
    inputLength = len(inp)
    trainDataSet = SupervisedDataSet(OBSERVE_LENGTH, PREDICT_LENGTH)
    for n in range(inputLength):
        if n + (OBSERVE_LENGTH - 1) + PREDICT_LENGTH < inputLength:
            trainDataSet.addSample(inp[n:n + OBSERVE_LENGTH], inp[n + 1:n + 1 + PREDICT_LENGTH])
    net = buildNetwork(OBSERVE_LENGTH, 20, PREDICT_LENGTH, outclass=LinearLayer, bias=True, recurrent=True)
    trainer = BackpropTrainer(net, trainDataSet)
    trainer.trainEpochs(100)
# endregion

# region Test
ts = UnsupervisedDataSet(OBSERVE_LENGTH, )
ts.addSample([21, 22, 23, 22, 21, 21, 22, 23, 23, 24])

print(int(net.activateOnDataset(ts)[0][0]))
# endregion

# region Save the network
netDumpFile = open(NETDUMP_FILENAME, 'wb')
pickle.dump(net, netDumpFile)
netDumpFile.close()
# endregion
