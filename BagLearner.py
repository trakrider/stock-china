"""
Bag Learner
"""
import numpy
import KNNLearner


class BagLearner(object):
    def __init__(self, learner = KNNLearner.KNNLearner, kwargs = {"k":3}, bags = 20, boost = False, verbose = False):
        self.learners = []
        self.bags = bags
        for i in range(bags):
            self.learners.append(learner(**kwargs))

    def addEvidence(self, dataX, dataY = None):
        for i in range(0,self.bags):
            m = len(dataY)                                 #ammount of training data randomized in each bag
            B = numpy.random.randint(len(dataX),size=m)    #random rows of data to be used in the bag
            trainingX = dataX[B,:]
            trainingY = dataY[B]
            self.learners[i].addEvidence(trainingX, trainingY)

    def query(self, dataX):
        dataY = numpy.zeros(dataX.shape[0])
        for learner in self.learners:
            dataY += learner.query(dataX)
        dataY = dataY/self.bags
        return dataY
