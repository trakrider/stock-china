# -*- coding: utf-8 -*-
"""
KNN Learner
"""

import numpy
from util import compute_euclidean_distance

class KNNLearner(object):

    def __init__(self, k = 3):
        self.k = k
        self.data = None
        self.dimensionX = 0

    def addEvidence(self, dataX, dataY = None):

        # Check X dimension
        if self.dimensionX == 0:
            self.dimensionX = dataX.shape[1]
        if not self.dimensionX == dataX.shape[1]:
            return

        # Create data to be inserted
        if not dataY == None:
            data = numpy.zeros([dataX.shape[0], dataX.shape[1]+1])
            data[:, 0:dataX.shape[1]] = dataX
            data[:, (dataX.shape[1])] = dataY
        else:
            return

        # Insert data
        if self.data is None:
            self.data = data
        else:
            self.data = numpy.append(self.data, data, axis = 0)

    def query_single_point(self, point, k = None):
        # Check input
        if numpy.isnan(point.sum()) == True:
            return None
        # Assign k
        if k is None:
            k = self.k

        # Compute distance
        train = numpy.zeros([self.data.shape[0], self.data.shape[1]+1])
        train[:, 1:self.data.shape[1]+1] = self.data
        for tp in train:
            tp[0] = compute_euclidean_distance(point, tp[1:], self.dimensionX)

        # Sort
        ide = numpy.argsort(train, axis = 0)    # index for sorted train

        # Generate output
        expect = 0
        for i in range(0,k):
            expect += train[ide[i][0], -1]        # Get Y value from sorted train
        expect = expect/k
        return expect

    def query(self, dataX):
        dataY = numpy.zeros(dataX.shape[0])
        for i in range(0, dataX.shape[0]):
            dataY[i] = self.query_single_point(dataX[i])

        return dataY


def test():
    pass

if __name__=="__main__":
    test()
