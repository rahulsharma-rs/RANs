__author__ = 'Rahul'

from Layer import Layer
#from RBM.utils import processInputFile
import numpy
import RAN_kfold as RAN
from UtilsRAN import processInputCSVFile,labelNodes,unSupervisedLabeling
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
#from owl_builder import build_rdf

def test_RAN():
        '''
        #creating RANs object
        iterError = None; stopCriteria=1
        error1=[]

        ran = RAN.RAN()


        #-------------------------------------------------------------------------------------------------------
        #path for the image data
        #path1 = '/home/rahul/turtle/mixedmnist'
        path1= '/home/rahul/turtle/newmnisttrain'
        #path2= '/Users/Rahul/PycharmProjects/turtle/yale/yaleB02'
        #path3= '/Users/Rahul/PycharmProjects/turtle/yale/yaleB03'
        path4= '/home/rahul/turtle/newmnisttest'
        #global variable from imgutils.py
        global dataSet, rows, columns
        ds =ts=None
        #storing the raw data for image
        dataSet,rows,columns=  prepImgDataSet(path1)
        ds=dataSet
        #dataSet,rows,columns=  prepImgDataSet(path2)
        #ds.extend(dataSet)
        #dataSet,rows,columns=  prepImgDataSet(path3)
        #ds.extend(dataSet)
        isConvex=True

        ds=numpy.asarray(ds)
        ran.rawData=ds
        ran.intermediateData.append(ran.rawData)
        '''

        # creating RANs object
        iterError = None;
        stopCriteria = 1
        error1 = []
        rows=28;columns=28
        ran = RAN.RAN()
        isConvex = True

        # data from .csv file
        path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/mnist1000data.csv"  # path of the file for training
        rawData, header = processInputCSVFile(path, normalize=True)  # input the
        trainD, sortedData, label, unique_label, labelFrequency = processData(data=rawData)
        # numpy.random.shuffle(rawData)
        # labels = rawData.T[rawData.T.__len__() - 1].T
        # croping label from the data
        ran.rawData = sortedData
        ran.intermediateData.append(ran.rawData)
        #training
        ran.train_RAN_Upward(ran,isConvex=isConvex,conceptIdentifier={"name":"k-mean","k":30})
        #ran.intermediateData1= ran.intermediateData
        ran.dataProjection=[None]*(ran.numberOfLayers)
        ran.intermediateTestData =[None]*(ran.numberOfLayers)
        #temp=[]
        ran.actualActivation =[[]]*(ran.numberOfLayers)
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
        ran.unique_labels, ran.label_frequency = numpy.unique(ran.labels, return_counts=True)
        ran.labelCount = numpy.unique(label).size
        ran.observation, ran.classLabels = labelNodes(RAN=ran)
        ran.startPoint[0] = ran.startPoint[0] * 0 + 1
        eAforcenters=[]
        for x in ran.weight[0]:
            eAforcenters.append(ran.propagateUpwardOld(ran.weight[0], x, isConvex=isConvex))

        #-testing with csv file----------------------------------------------------------------------------------
        #path1= "/Users/Rahul/Dropbox/RahulSharma/PHD/RAN_CRBMv1.0/data/test1.csv" # path of the file for training

        #ran.testData = processInputCSVFile(path1) # input the


        #displaying the centroid of the cluster obtained
        count =0
        for data in ran.weight[0]:
            path2= "/home/rahul/turtle/mixres1/img%s" %count
            path2 = path2+".png"
            imageData= arrayToImage(data, rows, columns)
            imageData = cv2.multiply(imageData,255) # de-normalizing the data to its original form
            #showImage(imageData,1)
            createImage(path2,imageData,1)
            count +=1
        stc=0

        while stc!=1:#loop until test inputs finishes
            testdata=[];tempx=[]
            stc2=1
            stc3=1
            reports=[]

            reports.append('-------RANs co-occurance-weights at layer-1 ------')
            for ets in ran.weightCoOccurance[1]:
                reports.append("%s" % ets)
            reports.append('--------Class-obeservations------------------------')
            for ets in ran.observation:
                reports.append("%s" % ets)
            numpy.savetxt("../report.txt", reports, delimiter="\n", fmt="%s")
            while stc3!=0:
                try:
                    print "total layer %s" %len(ran.weight)
                    cxx = float(raw_input('choose the layer for downward propagation : '))
                except ValueError:
                    stc3=1
                else:
                    if cxx>=0 and cxx <= len(ran.weight):
                        ran.startLayer= int(cxx)
                        stc3=0
                    else:
                        stc3=1
                        print "layer not in range"


            while stc2!=0:#loop unitl correct inputs are provided

                    try:
                        cx=float(raw_input('enter regulation factor [0-1]: '))


                    except ValueError:
                        stc2=1
                    else:
                        if cx<0 or cx>1:
                            stc2=1
                            print "enter value in range [0 1]\n"
                        else:
                            ran.rFactor=cx
                            stc2=0
            #for data1 in range(0, ran.intermediateData[len(ran.weight)].shape[1]):


            stci1=1
            while stci1 != 0:  # loop unitl correct inputs are provided
                try:
                    ch = float(raw_input("enter 1 to provide %s input seperated by comma"% ran.intermediateData[1].shape[1] +': '))


                except ValueError:
                    stci1 = 1
                else:
                    if ch < 0 or ch > 1:
                        stci1 = 1
                        print "enter value in range [0 1]\n"
                    else:
                        #testdata.append(c)
                        stci1 = 0
            if(ch==1):
                stcii1 = 1
                while stcii1 != 0:  # loop unitl correct inputs are provided
                    try:
                        cx = raw_input("enter the %s input(s) seperated by comma : "% ran.intermediateData[1].shape[1]).split(",")
                        c= [float(i) for i in cx]

                    except ValueError:
                        stcii1 = 1
                    else:
                        if c.__len__() != ran.intermediateData[1].shape[1]:
                            stcii1 = 1
                            print "enter values with %s data inputs\n"%ran.intermediateData[1].shape[1]
                        else:
                            testdata=c
                            stcii1 = 0
            else:

                for data1 in range(0, ran.intermediateData[ran.startLayer].shape[1]):
                    stc1=1
                    while stc1!=0:#loop unitl correct inputs are provided
                        try:
                            c=float(raw_input('enter value %s' %data1 +':'))


                        except ValueError:
                            stc1=1
                        else:
                            if c<0 or c>1:
                                stc1=1
                                print "enter value in range [0 1]\n"
                            else:
                                testdata.append(c)
                                stc1=0


            tempx.append(testdata)
            ran.intermediateTestData =[None]*(ran.numberOfLayers)
            ran.testData = numpy.asarray(tempx)
            #ran.intermediateTestData[len(ran.weight)]=ran.testData
            ran.intermediateTestData[ran.startLayer] = ran.testData
            ran.train_RAN_Downward(ran,ran.intermediateTestData)

        #----------------------------------------------------------------------------------------------------------




            #ct=0
            tempx1=[]
            for x1 in ran.dataProjection[0]:
                if x1.min()<0 or x1.max()>1:
                    tempx1.append(ran.normalizeData(x1))
                else:
                    tempx1.append(x1)
            temx=numpy.asarray(tempx1)


            #ran.dataProjection[0]=5*abs(ran.dataProjection[0])
            tmpx=[]


            count=0
            for data in temx:
            #for data in ran.dataProjection[0]:
            #for data in ran.intermediateDataObtained[0]:
                path2= "/home/rahul/turtle/mixres/img%s" %count
                path2 = path2+".png"
                #path3= "/Users/Rahul/PycharmProjects/turtle/testmnistcent/img%s" %ct
                #path3 = path3+".png"
                imageData= arrayToImage(data, rows, columns)
                imageData = cv2.multiply(imageData,255) # de-normalizing the data to its original form
                #showImage(imageData,300)

                createImage(path2,imageData,1)
                count +=1 #;ct+=1
            stop=1
            while stop==1:
                try:
                    stc=int(raw_input('Enter 1 to stop iteration: '))
                    stop=0
                except ValueError:
                    stop = 1





        '''
        ct1=ct2=ct3=miss=0

        for x in range(0,149):
            if x<50 and ran.intermediateData[6][x][0]==ran.intermediateData[6][x].max():
                ct1+=1
            elif x>=50 and x<100 and ran.intermediateData[6][x][1]==ran.intermediateData[6][x].max():
                ct2+=1
            elif x>=100 and  ran.intermediateData[6][x][2]==ran.intermediateData[6][x].max():
                ct3+=1
            else:
                miss+=1
        '''
        '''
        #---------------------plotting parallel coordinates--------------------------------------------------------
        filectr=0

        plt.figure(1)
        #i=0
        #ran.dproj=numpy.transpose(ran.dataProjection[0]).tolist()
        #dproj=numpy.transpose(ran.dataProjection[0])
        #plt.plot(dproj[0],dproj[1],label='$y = {i}x + {i}$'.format(i=i))

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

        ran.rFactor=0
        #data=ran.regulate()
        rdata=numpy.transpose(ran.rawData)
        centroids=numpy.transpose(ran.weight[0])
        #centers=ran.weight[0]

        for color in ['r']:#, 'b']:#, 'g', 'k', 'm','c','y']:#,'brown','darkgreen','purple']:
            #file='ober%s.csv' %filectr
            l='Activation Propagation'
            #l="%s%%  " %(ran.rFactor*100) + "regulation"
            ran.testNplot1(ran,colour=color,path1=path1,label=l)
            ran.rFactor+= 0.075
            filectr+=1
        dataproj= numpy.transpose(ran.dataProjection[0])

        plt.plot(rdata[0],rdata[1],'yo')
        ctr1=0
        centerColors={1:'ko',0.90:'#A52A2A',0.80:'#D2691E',0.70:'#006400',0.60:'#BDB76B',0.50:'#00008B',0.40:'#00008B',0.30:'#8B008B',0.20:'#FF8C00',0.10:'#8B0000',0.0:'#FF1493'}
        for x in numpy.transpose(centroids):
            val=ran.intermediateTestData[1][0][ctr1]
            #val=centerColors[round(val,1)]
            #val=str(val)
            plt.plot(x[0],x[1],color='green', linestyle='dashed', marker='o',
     markerfacecolor='black', markersize=(5+8*val))
            ctr1+=1

        #plt.plot(rdata[0],rdata[1],'yo',label='Data-points',marker='o')
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
        '''

        print 'this mesg'
        #print ran.maximumActivationCountForEachTopLayerNode
if __name__ == "__main__":
    test_RAN()