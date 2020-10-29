__author__ = 'Rahul'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 RBM  w/ continuous-valued inputs (Linear Energy)
 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""

import sys
import numpy
from RBM import RBM
from utils import *
from Kmean import *
from Kmean import Point
from Kmean import Cluster
import csv
from UtilsRAN import processInputCSVFile


class CRBM(RBM):
    def propdown(self, h):
        pre_activation = numpy.dot(h, self.W.T) + self.vbias
        return pre_activation



    def sample_v_given_h(self, h0_sample):
        a_h = self.propdown(h0_sample)
        en = numpy.exp(-a_h)
        ep = numpy.exp(a_h)

        v1_mean = 1 / (1 - en) - 1 / a_h
        U = numpy.array(self.numpy_rng.uniform(
            low=0,
            high=1,
            size=v1_mean.shape))

        v1_sample = numpy.log((1 - U * (1 - ep))) / a_h

        return [v1_mean, v1_sample]



def test_crbm(learning_rate=0.1, k=1, training_epochs=1000):
    '''data = numpy.array([[0.4, 0.5, 0.5, 0.,  0.,  0.],
                       [0.5, 0.3,  0.5, 0.,  0.,  0.],
                      [0.4, 0.5, 0.5, 0.,  0.,  0.],
                      [0.,  0.,  0.5, 0.3, 0.5, 0.],
                     [0.,  0.,  0.5, 0.4, 0.5, 0.],
                       [0.,  0.,  0.5, 0.5, 0.5, 0.]])  '''

    reader=csv.reader(open("test.csv","rb"),delimiter=',')
    x=list(reader)
    data=numpy.array(x).astype('float')
    path= "/Users/Rahul/Dropbox/RahulSharma/PHD/RAN_CRBMv1.0/data/iris.csv" # path of the file for training

    rawData = processInputCSVFile(path)
    '''data = numpy.array([[0.46, 0.51, 0.49, 0.01,  0.02,  0.001],
                       [0.38, 0.49,  0.51, 0.03,  0.,  0.01],
                      [0.4, 0.5, 0.5, 0.,  0.,  0.],
                       [0.39,  0.48,  0.48, 0.04, 0.001, 0.001]])'''


    rng = numpy.random.RandomState(123)

    # construct CRBM
    rbm = CRBM(input=data, n_visible=6, n_hidden=5, numpy_rng=rng)



    # train
    for epoch in xrange(training_epochs):

        ph_sample= rbm.contrastive_divergence(lr=learning_rate, k=k)
        # cost = rbm.get_reconstruction_cross_entropy()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost

    #ph_mean, ph_sample = rbm.sample_h_given_v(data)
    # test
    #mean, vSample= rbm.sample_v_given_h(ph_sample)
    mean, hsample=rbm.sample_h_given_v(data)
    temp=[tuple(row) for row in ph_sample]
    hsampleunique= numpy.unique(temp)
    ph_sample= numpy.ndarray.tolist(ph_sample)
    print hsampleunique
    centroids=[]
    cluster=[]

    for x in hsampleunique:
        temp=[]
        for y in data:
            i, temp1= rbm.sample_h_given_v(y)
            if all((temp1== x)):
                temp.append(y)
        cluster.append(temp)
    cluster = numpy.asarray(cluster)
    for clus in cluster:
        count=0
        lenght=clus.shape[1]
        tempx=numpy.zeros(lenght)
        for j in clus:
            tempx = numpy.add(tempx,j)
            count+=1
        tempx = numpy.divide(tempx,count)
        centroids.append(tempx)
    #print ph_sample

    i, temp1= rbm.sample_v_given_h([0, 1 ,1 ,0 ,1])
    print temp1
    print centroids
    print ph_sample
    '''data = numpy.array([[0.46, 0.51, 0.49, 0.01,  0.02,  0.001],
                       [0.38, 0.49,  0.51, 0.03,  0.,  0.01],
                      [0.4, 0.5, 0.5, 0.,  0.,  0.],
                       [0.39,  0.48,  0.48, 0.04, 0.001, 0.001]])'''


    # test
    #ph_sample= numpy.ndarray.tolist(data)
    #print ph_sample
    clusters= None
    points=[]
    '''
    for i in ph_sample:

            points.append(Point(i))

    clusters = kmeans(points, 3, 0.5)

    for i,c in enumerate(clusters):
        x= c.calculateCentroid()
        for p in c.points:

            print " Cluster: ", i, "\t Point :", p, "centroid:", x

    # Display clusters using plotly for 2d data
    # This uses the 'open' command on a URL and may only work on OSX.
    # test
    #v = numpy.array([[0.46, 0.51, 0.49, 0.01,  0.02,  0.001],
       #                [0.38, 0.49,  0.51, 0.03,  0.,  0.01]])
    '''
    #print rbm.reconstruct(v)


if __name__ == "__main__":
    test_crbm()