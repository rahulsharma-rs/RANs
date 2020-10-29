__author__ = 'Rahul'
''' '''
import numpy
import csv

numpy.seterr(all='ignore')


def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))


def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

#method to process the input file in csv format, and return and numpy array and count of the inputs
def processInputFile(filePath):

    reader=csv.reader(open(filePath,"rb"),delimiter=',')
    x=list(reader)
    data=numpy.array(x).astype('float')

    return data


# # probability density for the Gaussian dist
# def gaussian(x, mean=0.0, scale=1.0):
#     s = 2 * numpy.power(scale, 2)
#     e = numpy.exp( - numpy.power((x - mean), 2) / s )

#     return e / numpy.square(numpy.pi * s)