from __future__ import division
__author__ = 'Rahul'
import numpy
import csv
import matplotlib.pyplot as plt
import itertools
import warnings
import matplotlib

from scipy._lib._util import _asarray_validated
from scipy._lib import _numpy_compat




# from __future__ import division it makes all division '/' operation as floating point, to make integer division use '//'

"""
method to process the input CSV file in format
first row should represents the maximum value represented by each column (feature), from second row onwards the CSV file.
the method returns a matrix of data normalized between [0,1]

----------------------------------------------------
max-value-col1 , max-value-col2, max-value-col3, ...
----------------------------------------------------



------------------------------------------------------
Usage:

<return_variable> = processInputCSVFile(filePath=<path_of_file>)
    by default:
        filePath=None
"""
def processInputCSVFile(filePath=None,normalize=True):

    reader=csv.reader(open(filePath,"r"),delimiter=',')
    x=list(reader)
    header= numpy.asarray(x[0]).astype('float') # first line extracted that holds the maximum values for each category
    normalizer = 1/header
    data=temp=numpy.asarray(x[1:]).astype('float') #from second line onwards the actual data

    if normalize==True:
        data = temp*(normalizer)
    #data= numpy.around(data, decimals=2)
    return data,header

"""
This method split and returns the provided data into test and training data
Usage:
<train_data>, <test_data> = split_data(data=None, size=0.40)

    by default:
            data is None and size of test_set is 40% of the provided data

"""
def split_data(data=None,size=0.40 ):

    data_size= data.shape[0]
    train_data= data[:int(data_size*size)]
    test_data = data[int(data_size*size+1):]

    return [train_data, test_data]


def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+numpy.exp(-z))


"""
This method associate a Label to a top layer Nodes based upon the training data
"""
def labelNodes(RAN=None):
    #data is the training data obtained in the top layer
    #labels are the training lables
    temp = []

    ctr = 0
    start = 0
    for x in RAN.label_frequency:

        # ct =[[0]*(ran.intermediateData1[ran.intermediateData1.__len__()-1].shape[1])+[0]]
        ct = numpy.asarray([[0.0] * (RAN.intermediateData[RAN.intermediateData.__len__() - 1].shape[1]) + [0.0]])  #
        ct = numpy.squeeze(ct)

        # ct=numpy.zeros(ran.intermediateData1[ran.intermediateData1.__len__()-1]+1)#
        for y in range(0, x):
            i = RAN.intermediateData.__len__() - 1
            j = start
            k = RAN.intermediateData[i][j]
            l = RAN.intermediateData[i][j].argmax()
            ct[RAN.intermediateData[RAN.intermediateData.__len__() - 1][start].argmax()] += 1
            start += 1

        # start=x
        ##intermediate test varibalmes
        ctt = numpy.divide(ct, x)
        ctt = numpy.divide((ctt - ctt.min()), (ctt.max() - ctt.min()))
        ctt1 = numpy.divide(ct, x)
        ###-----variable ends
        ct[ct.size - 1] = RAN.unique_labels[ctr]
        ctr += 1
        temp.append(ct)
    temp=numpy.asarray(temp)
    #label = numpy.zeros(RAN.labelCount-1)
    label=numpy.zeros(temp.shape[0]) #label->node_ID_of_top_layer_concept example label[0]=3 mean label '0' in data points to node 3 in top layer
    label=label*(-1)
    for i in temp:
        xi=int(i[i.size-1]*(RAN.labelCount-1))
        label[int(i[i.size-1]*(RAN.labelCount-1))]=numpy.argmax(i)

    return temp,label

def unSupervisedLabeling(RAN=None):
    data=RAN.intermediateData[RAN.numberOfLayers-1]
    label=[]
    activation=[]
    for dta in data:
        label.append(numpy.argmax(dta))
        activation.append(dta[numpy.argmax(dta)])
    #print ('USL')
    return label,activation

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0,fontsize=7)
    plt.yticks(tick_marks, classes,rotation=90,fontsize=7)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #



    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=6)
    plt.xlabel('Predicted Label',fontsize=6)

def whiten(obs, check_finite=True):
    """
    Normalize a group of observations on a per feature basis.
    Before running k-means, it is beneficial to rescale each feature
    dimension of the observation set with whitening. Each feature is
    divided by its standard deviation across all observations to give
    it unit variance.
    Parameters
    ----------
    obs : ndarray
        Each row of the array is an observation.  The
        columns are the features seen during each observation.
        >>> #         f0    f1    f2
        >>> obs = [[  1.,   1.,   1.],  #o0
        ...        [  2.,   2.,   2.],  #o1
        ...        [  3.,   3.,   3.],  #o2
        ...        [  4.,   4.,   4.]]  #o3
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    Returns
    -------
    result : ndarray
        Contains the values in `obs` scaled by the standard deviation
        of each column.
    Examples
    --------
    >>> from scipy.cluster.vq import whiten
    >>> features  = numpy.array([[1.9, 2.3, 1.7],
    ...                       [1.5, 2.5, 2.2],
    ...                       [0.8, 0.6, 1.7,]])
    >>> whiten(features)
    array([[ 4.17944278,  2.69811351,  7.21248917],
           [ 3.29956009,  2.93273208,  9.33380951],
           [ 1.75976538,  0.7038557 ,  7.21248917]])
    """
    obs = _asarray_validated(obs, check_finite=check_finite)
    std_dev = numpy.std(obs, axis=0)
    zero_std_mask = std_dev == 0
    if zero_std_mask.any():
        std_dev[zero_std_mask] = 1.0
        warnings.warn("Some columns have standard deviation zero. "
                      "The values of these columns will not change.",
                      RuntimeWarning)
    return (obs / std_dev), std_dev

def plotROC(fpr=None,tpr=None,path=None,auc=None):
    plt.figure()

    #ax=plt.subplot(121)
    lw = 2
    #color = ['b', 'g', 'r', 'c', 'k', ]\
    skip=['gold','yellow','ivory','greenyellow','white','floralwhite','aliceblue','ghostwhite','lavander','honeydes','w','whitesmoke','snow','lemonchiffon','azure','linen',
          'antiquewhite','papayawhip','oldlace','cornsilk','palegoldenrod','lightyellow','mintcream','lightcyan','lavenderblush']
    color = []
    for cname, cvalue in matplotlib.colors.cnames.items():
        if cname in skip:
            continue
        else:
            color.append(cname)

    ctr=0
    for i in range(0,len(fpr)):
        plt.plot(fpr[i], tpr[i], color=color[i],
             lw=lw, label='Class-%0.0f (AUC = %0.2f%%)' % (i,auc[i]*100.00))
        ctr+=1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.05, 0.6), loc=1, borderaxespad=0.,prop={'size':10})
    #lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
    plt.savefig(path)

    #plt.show()
    plt.close()