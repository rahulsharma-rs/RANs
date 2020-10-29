__author__ = 'Rahul'

from Layer import Layer
#from RBM.utils import processInputFile
import numpy
from UtilsRAN import processInputCSVFile,labelNodes
from scipy.spatial import distance
from mnist import *
import cv2
import os
import statistics
import matplotlib.pyplot as plt
from imgutil import *
from sklearn import cluster
import pandas
from pandas.tools.plotting import parallel_coordinates
from Processing_split_data import processData
import matplotlib as mpl
import csv
import RAN_kfold as RAN

def test_RAN():

        #creating RANs object
        iterError = None; stopCriteria=1
        error1=[]

        ran = RAN.RAN()
        isConvex=True

        #data from .csv file
        path= "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/Toy-dataRANs4testclusterprep.csv" # path of the file for training
        rawData, header = processInputCSVFile(path,normalize=True) # input the
        trainD, sortedData, label, unique_label, labelFrequency = processData(data=rawData)
        #numpy.random.shuffle(rawData)
        #labels = rawData.T[rawData.T.__len__() - 1].T
        # croping label from the data
        ran.rawData = sortedData
        ran.intermediateData.append(ran.rawData)
        #path1="/Users/Rahul/Dropbox/RahulSharma/PHD/RAN_CRBMv1.0/RBM/test1.csv"
        #ran.testData=processInputFile(path1)
        #ran.intermediateTestData.append(ran.testData)

        #--------------------------------------------------------------------------------------------------------

        #training
        ran.train_RAN_Upward(ran,isConvex=isConvex,conceptIdentifier={"name":"k-mean","k":5})
        #ran.intermediateData1= ran.intermediateData
        ran.dataProjection=[None]*(ran.numberOfLayers)
        ran.intermediateTestData =[None]*(ran.numberOfLayers)
        #temp=[]
        ran.actualActivation =[[]]*(ran.numberOfLayers)
        ran.error1=[[]]*(ran.numberOfLayers)
        for dt in ran.intermediateData:
            ran.startPoint.append(numpy.random.rand(dt.shape[1]))

        tempw=[]
        if ran.weight[0].min()<0 or ran.weight[0].max()>1:
            #ran.weight[0]= 2*ran.sigmoid(ran.weight[0])-1
            for x in ran.weight[0]:
                if x.min()<0 or x.max()>1:
                    tempw.append(ran.normalizeData(x))
                else:
                    tempw.append(x)
            ran.weight[0]=numpy.asarray(tempw)
            #ran.weight[0]=ran.normalizeData(ran.weight[0])
        #--------------------------------------------------------------------------------------------------------
        for w in ran.weight:

            ran.weightDown.append(numpy.transpose(w))

        #--------------------------------------------------------------------------------------------------------
        ran.labels = label
        ran.unique_labels, ran.label_frequency = numpy.unique(ran.labels,return_counts=True)
        ran.labelCount = numpy.unique(label).size
        ran.observation, ran.classLabels = labelNodes(RAN=ran)
        #-testing with csv file----------------------------------------------------------------------------------
        path1= "/home/rahul/PycharmProjects/RAN_CRBMv1.0/data/test1.csv" # path of the file for training
        #ran.testData = processInputCSVFile(path1) # input the
        #ran.intermediateTestData[1]=ran.testData

        #ran.test(ran)
        #ran.train_RAN_Downward(ran,ran.intermediateTestData)

        #---------------------------------------------------------------------------------------------------------
        ran.startPoint[0][0]=0
        ran.startPoint[0][1]=0.6
        stcx=0
        while stcx!=1:
            aA=[];err=[];reports=[]


            reports.append('-------RANs co-occurance-weights at layer-0 ------')
            for ets in ran.weightCoOccurance[0]:
                reports.append("%s" % ets)
            reports.append('-------RANs co-occurance-weights at layer-1 ------')
            for ets in ran.weightCoOccurance[1]:
                reports.append("%s" % ets)
            reports.append('--------Class-labels------------------------')
            ctrClass=0
            for x in ran.classLabels:
                str='Class-%s is -> %s'%(ctrClass,x)
                ctrClass+=1
                reports.append(str)

            reports.append('--------------------------------')

            fif, ax = plt.subplots()
            ax.set_ylim(-0.2, 1.2)
            ax.set_xlim(-0.2, 1.2)
            pdxt = pandas.read_csv("/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/Toy-dataRANs4testclusterprepRaw.csv")
            pdxt.plot(ax=ax, kind='scatter', x='Dimension-x', y='Dimension-y', c='labels', colormap='Dark2')
            center = pandas.DataFrame(ran.weight[0], columns=['Dimension-X', 'Dimension-Y'])
            center.plot(ax=ax, x='Dimension-X', y='Dimension-Y', kind='scatter', c='Yellow', s=75)
            temp2 = ['Regulation'] + ['--'] + ['        Expected Activation        '] +['--']+ ['                 Observed Activation                   '] + ['--'] + ['Observed Activation At Input Layer'] + ['--']+['Figure']
            temp3 = None
            ctr = 0
            for x in temp2:
                if ctr == 0:
                    temp3 = "%s" % (x)
                    ctr = 1
                else:
                    temp3 = "%s\t%s" % (temp3, x)
            reports.append(temp3)
            reports.append('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            stc=0
            figCtr=0
            while stc!=1:#loop until test inputs finishes
                testdata=[];tempx=[]
                stc2=1
                while stc2!=0:#loop unitl correct inputs are provided
                        try:
                            cx=float(raw_input('Enter regulation factor [0-1]: '))


                        except ValueError:
                            stc2=1
                        else:
                            if cx<0 or cx>1:
                                stc2=1
                                print "Kindly enter the value in range [0 1]\n"
                            else:
                                ran.rFactor=cx
                                stc2=0
                '''
                stc3=1
                while stc3!=0:#loop unitl correct inputs are provided
                        try:
                            cx1=float(raw_input('enter the influncer of Impact 2 or 3: '))
                        except ValueError:
                            stc3=1
                        else:
                            if cx1==2 or cx1==3:
                                ran.impactInfluencer = cx1
                                stc3 = 0
                            else:
                                stc3 = 1
                                print "enter either 2 or 3\n"
                '''

                for data1 in range(0, ran.intermediateData[1].shape[1]):
                    stc1=1
                    while stc1!=0:#loop unitl correct inputs are provided
                        try:
                            c=float(raw_input('Enter Expected Activation value as Node-%s' %data1 +':'))


                        except ValueError:
                            stc1=1
                        else:
                            if c<0 or c>1:
                                stc1=1
                                print "Kindly enter the value in range [0 1]\n"
                            else:
                                testdata.append(c)
                                stc1=0


                tempx.append(testdata)
                ran.intermediateTestData =[None]*(ran.numberOfLayers)
                ran.testData = numpy.asarray(tempx)
                ran.intermediateTestData[1]=ran.testData
                ran.train_RAN_Downward(ran,ran.intermediateTestData)
                ran.intermediateData1[0]=numpy.asarray([ran.dataProjection[0][1001]])

                #actualActivation1 = ran.propagateUp(ran,isConvex=isConvex)
                #actualActivation = ran.propagateUpward(ran.weight[0], ran.dataProjection[0][1001], isConvex=isConvex)
                actualActivation2 = ran.propagateUpwardOld(ran.weight[0], ran.dataProjection[0][1001], isConvex=isConvex)

                #reports.append("------ Regulatio Factor %s%%----"%(ran.rFactor*100))
                projD1 = pandas.DataFrame(['reg-0'] * ran.dataProjection[0].shape[0], columns=['label'])
                projD = pandas.DataFrame(ran.dataProjection[0], columns=['Dimension-X', 'Dimension-Y'])
                px = pandas.DataFrame({'Dimension-X': [ran.dataProjection[0][10001][0]], 'Dimension-Y': [ran.dataProjection[0][10001][1]]})
                px1 = pandas.DataFrame({'Dimension-X': [ran.dataProjection[0][0][0]], 'Dimension-Y': [ran.dataProjection[0][0][1]]})
                px.plot(ax=ax, x='Dimension-X', y='Dimension-Y', kind='scatter', c='Black', s=30)
                px1.plot(ax=ax, x='Dimension-X', y='Dimension-Y', kind='scatter', c='Red', s=30)
                regLabel="reg-%s%%" %(ran.rFactor*100)
                temp=[regLabel]+['--']+testdata+['--']+(numpy.around(actualActivation2,decimals=7)).tolist()+['--']+(numpy.around(ran.dataProjection[0][10001],decimals=7)).tolist()+['--']
                temp1=None
                ctr=0
                for x in temp:
                    if ctr==0:
                        temp1="%s"%(x)
                        ctr=1
                    else:
                        temp1="%s\t%s"%(temp1,x)

                projD.plot(ax=ax, x='Dimension-X', y='Dimension-Y', label=regLabel)
                ax.set_ylim(-0.2, 1.2)
                ax.set_xlim(-0.2, 1.2)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True,
                          prop={'size': 10})

                fig="Fig-%s"%figCtr
                figCtr+=1
                temp1 = "%s\t%s" % (temp1, fig)
                reports.append(temp1)
                fif.savefig("../%s.png" %fig)
                #----------------------------------------------------------------------------------------------------------


                stop=1
                while stop==1:
                    try:
                        stc=int(raw_input('Enter 1 to stop iteration'))
                        stop=0
                    except ValueError:
                        stop = 1
            stopx = 1
            while stopx==1:
                try:
                    stcx = int(raw_input('Enter 1 to stop iteration for map'))
                    stopx = 0
                    numpy.savetxt("../report.txt", reports, delimiter="\n", fmt="%s")
                except ValueError:
                    stopx = 1




        #---------------------plotting parallel coordinates--------------------------------------------------------
        filectr=0

        plt.figure(1)
        #i=0
        #ran.dproj=numpy.transpose(ran.dataProjection[0]).tolist()
        #dproj=numpy.transpose(ran.dataProjection[0])
        #plt.plot(dproj[0],dproj[1],label='$y = {i}x + {i}$'.format(i=i))
        '''
        for color in ['r', 'b', 'g', 'k', 'm','c','y']:#,'brown','darkgreen','purple']:
            file='ober%s.csv' %filectr
            ran.testNplot(ran,colour=color,path1=path1,file=file,label='ober%s' %filectr)
            ran.rFactor+= 0.02
            filectr+=1
            #ran.testNplot(ran,path1=path1,x='ro')

        #ran.test(ran)
        #i+=1

        #ran.dproj[0]=ran.dproj[0]+dproj[0]
        #ran.dproj[1]=ran.dproj[1]+dproj[1]
        #--------------------------------------------------------------------------------------------------------------


        d1=pandas.read_csv("../data/t13wl.csv")
        parallel_coordinates(d1,'label')


        ran.dataconversion(ran.weight[0],"../data/centroids.csv",ran.weight[0].shape[1],label="Centeroids")
        d1=pandas.read_csv("../data/centroids.csv")
        parallel_coordinates(d1,'label',color='k')

        #plt.show()'

       #---------------------plotting scatter 2-d plot-------------
        plt.figure(2)
        '''
        ran.rFactor=0
        #data=ran.regulate()
        rdata=numpy.transpose(ran.rawData)
        centroids=numpy.transpose(ran.weight[0])
        #centers=ran.weight[0]

        for color in ['r', 'b']:#, 'g', 'k', 'm','c','y']:#,'brown','darkgreen','purple']:
            #file='ober%s.csv' %filectr
            #l='Activation Propagation'
            l="%s%%  " %(ran.rFactor*100) + "regulation"
            ran.testNplot1(ran,colour=color,path1=path1,label=l)
            ran.rFactor+= 0.075
            temppp= ran.dataProjection[0][1001]
            filectr+=1
        dataproj= numpy.transpose(ran.dataProjection[0])

        #plt.plot(rdata[0],rdata[1],'yo')
        plt.plot(rdata[0], rdata[1], 'yo', label='Raw-Data', marker='o')
        ctr1=0
        centerColors={1:'ko',0.90:'#A52A2A',0.80:'#D2691E',0.70:'#006400',0.60:'#BDB76B',0.50:'#00008B',0.40:'#00008B',0.30:'#8B008B',0.20:'#FF8C00',0.10:'#8B0000',0.0:'#FF1493'}
        for x in numpy.transpose(centroids):
            val=ran.intermediateTestData[1][0][ctr1]
            #val=centerColors[round(val,1)]
            #val=str(val)
            plt.plot(x[0],x[1],color='green', linestyle='dashed', marker='o',
     markerfacecolor='black', markersize=(5+8*val))
            ctr1+=1
        #plt.legend()
        #plt.show()
        #plt.figure(2)

        #plt.plot(centroids[0],centroids[1], 'ko',label='Centroids',marker='o')
        plt.legend()
        #plt.xlabel("Dimension 0")
        #plt.ylabel("Dimension 1")
        #plt.plot(dataproj[0],dataproj[1], 'b')
        #dproj=numpy.asarray(ran.dproj)
        #plt.plot(dproj[0],dproj[1],'b')
        #for x in ran.cCenter:
           # plt.plot(x[0],x[1],'ko')

        plt.show()


        print 'this mesg'
        #print ran.maximumActivationCountForEachTopLayerNode
if __name__ == "__main__":
    test_RAN()