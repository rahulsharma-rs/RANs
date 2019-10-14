__author__ = 'Rahul'

#from RBM.CRBM import CRBM
import numpy
from sklearn import cluster
import time
from simpleClustering import sCluster
class Layer(object):

    def __init__(self, data=None,numberOfInputs=None, id=None,isconvex=True, objRAN=None):
        self.nCWeight=None#weights learned from non-convex layer
        self.coOccurMatrix = numpy.zeros((numberOfInputs, numberOfInputs))  # initializing the courrance matrixc
        self.coOccurMatrix1=numpy.zeros((numberOfInputs, numberOfInputs))
        self.individualOccuranceCount = numpy.zeros(
            (numberOfInputs))  # initializing the counter for individual activation count
        self.InterLayerWeightAbove = None  # Weight Matrix for Above Layer
        self.InterLayerWeightBelow = None  # Weight matrix for layer below

        self.inputActivation = None  # for interLayer and intraLayer operation

        self.outputActivation = None  # propagating output to lower layer or to the output

        self.phSample = None  # sample of hidden nodes

        self.numberOfCertroidIdentified = None
        self.numberOfNonConvexNodes= None

        self.trainDataToAbovelayer = None
        self.crbm = None
        self.cluster = None
        if isconvex==True:
            if id != None:
                self.layerID= id
            if data != None:
                self.inputTrainData= data
                self.numberOfVisibleNodes= numberOfInputs
                #if self.numberOfVisibleNodes>=100:
                 #   self.numberOfHiddenNodes=
                #else:
                self.numberOfHiddenNodes = self.numberOfVisibleNodes-1# keepinf the number of hidden nodes one less than that of visible nodes
                #if self.numberOfHiddenNodes ==1:
                 #   self.numberOfHiddenNodes=2
            # in case new layer is created without new data only number of input nodes
            elif data == None and numberOfInputs !=None:
                self.numberOfVisibleNodes= numberOfInputs
                self.numberOfHiddenNodes = self.numberOfVisibleNodes-1
                #if self.numberOfHiddenNodes ==1:
                 #   self.numberOfHiddenNodes=2

        #initialization and assignments for non-convex concepts layer
        if isconvex==False:
            print " *** Non-convex concepts***"
            wt_std=[];wt_avg=[];wt_var=[]# variable for weightd sandard deviation, average, and variance
            threshold=0.85
            if id!=None:
                self.layerID=id
            if data!=None:
                self.inputTrainData=data
                #logic for observing co-rrelation matrix for merging the convex nodes to make non-convex nodes and learn theirs associations
                ls = objRAN.weightCoOccurance[id]
                for ti in ls:
                    av,st= self.weighted_avg_and_std(values=ti,weights=ti)
                    wt_avg.append(av);wt_std.append(st)
                wt_var=numpy.square(wt_std)
                tp_tres1=numpy.average(wt_avg-wt_var)+numpy.average(wt_std)
                #tp_tres2=numpy.average(wt_avg+wt_var)+numpy.average(wt_std)
                #tp_tres3 = numpy.average(wt_avg - wt_var) - numpy.average(wt_std)
                #tp_tres3 = numpy.average(wt_avg + wt_var) - numpy.average(wt_std)
                #if tp_tres1>threshold and tp_tres1<1.0 and
                l = []
                coIndex = []
                emptyCoIndex=[]
                wt = []
                ctr=0
                for x in ls:
                    tem = []
                    t = []
                    for y in range(0, x.size):
                        if x[y] >= 0.70:#0.776749234944:#tp_tres1:#
                            t.append(1)
                            #t.append(x[y])#CHANGED TO VARYING WEIGHTS INSTEAD OF 1'S
                            tem.append(y)
                        else:
                            t.append(0)
                    #if not tem:
                        #emptyCoIndex.append(ctr)
                    #else:
                    coIndex.append(tem)
                    ctr+=1
                    l.append(t)
                #coIndex=coIndex+[emptyCoIndex]
                l = numpy.asarray(l)
                checkedNodes = []
                nodeCount = 0

                for i in range(0, ls.shape[1]):

                    if i in checkedNodes:
                        continue
                    else:
                        w = numpy.zeros(ls.shape[1])#l[i]
                        '''
                        for j in coIndex[i]:
                            w = w | l[j]
                        '''
                        tmp = []
                        tmp.append(i)
                        tmp = tmp + coIndex[i]
                        w[tmp]=1
                        '''
                        #UNCOMENT TO HAVE VARYING WEIGHTS INSTEAD OF 1's
                        w[i] = 1  #
                        for i1 in coIndex[i]:
                            w[i1] = ls[i][i1]
                        '''
                        nodeCount += 1
                        checkedNodes = checkedNodes+tmp#numpy.unique(numpy.asarray(checkedNodes + tmp)).tolist()
                        wt.append(w)
                self.numberOfNonConvexNodes =self.numberOfVisibleNodes= nodeCount
                self.numberOfHiddenNodes = self.numberOfVisibleNodes - 1
                self.nCWeight=numpy.asarray(wt)
                #self.coOccur(self.inputTrainData)  # calling to make co-occurance matrix
                print "constructor of non-convex concept at layer ", id

    #Method to perform Layer operation
    def layerOperation(self,layerInfo,isConvex=False, conceptIdentifier=None):

        if self.inputTrainData != None:

            if conceptIdentifier["name"]=="rbm":


                #layer operation using CRBM
                learning_rate=0.00101; k=1; training_epochs=100
                if layerInfo>1:
                    learning_rate=0.0001
                rng = numpy.random.RandomState(12345)

                #constricting the RBM
                self.crbm = CRBM(input=self.inputTrainData, n_visible=self.numberOfVisibleNodes, n_hidden=self.numberOfHiddenNodes, numpy_rng=rng)

                # train
                for epoch in xrange(training_epochs):
                    self.phSample= self.crbm.contrastive_divergence(lr=learning_rate, k=k)
                self.prepareCentroidMatrix(self.phSample,self.crbm)

            elif conceptIdentifier["name"]=="meanshift":

                #layer operation with meanshift
                bandWidth=cluster.estimate_bandwidth(self.inputTrainData,quantile=0.2,n_samples=500)
                ms=cluster.MeanShift(bandwidth=bandWidth)
                ms.fit(self.inputTrainData)
                self.InterLayerWeightAbove=ms.cluster_centers_
                self.numberOfCertroidIdentified =ms.cluster_centers_.shape[0]

            elif conceptIdentifier["name"]=="sc":
                #simple clustering
                clusters,self.InterLayerWeightAbove=sCluster(self.inputTrainData)
                self.numberOfCertroidIdentified=self.InterLayerWeightAbove.__len__()
            elif conceptIdentifier["name"]=="k-mean":
                #layer operation using k-mean
                if layerInfo==1:
                    n_of_clusters=conceptIdentifier["k"]
                elif layerInfo==2:
                    n_of_clusters=1
                elif layerInfo==3:
                    n_of_clusters=1
                else:
                    n_of_clusters=1
                k_mean=cluster.KMeans(n_of_clusters,max_iter=1000)
                k_mean.fit(self.inputTrainData)
                self.InterLayerWeightAbove=k_mean.cluster_centers_
                self.numberOfCertroidIdentified=n_of_clusters
            elif conceptIdentifier["name"]=="afp":


                #layer operation using affinity clustering
                if layerInfo>3:
                    #dampF=0.9679
                    dampF=0.9679
                else:
                    dampF=0.94
                    #dampF=.9679
                affine=cluster.AffinityPropagation(damping=dampF,convergence_iter=15,max_iter=1000)
                affine.fit(self.inputTrainData)
                self.InterLayerWeightAbove= affine.cluster_centers_
                self.numberOfCertroidIdentified= self.InterLayerWeightAbove.shape[0]

            else:

                print "Unidentified Concept Identifier"
                return

            #------------------------------------------------
            #courrance matrix formation
            #------------------------------------------------
            #if isConvex==False:
            

            st_time = time.clock()
            self.coOccurNew(self.inputTrainData)
            print time.clock() - st_time, "seconds"

        print "Operation for each layer"
        return self.InterLayerWeightAbove, self.numberOfCertroidIdentified, self.coOccurMatrix, self.individualOccuranceCount

    #new co-occurnace matrix
    def coOccurNew(self,data):
        ctrx=0
        ctry=0
        d=numpy.transpose(data)
        for x in d:
            for y in range(ctrx,d.shape[0]):
                if y==ctrx:
                    self.coOccurMatrix[ctrx][y]=0

                else:
                    tempN=(1-numpy.abs(x-d[y])-0.5*(1-x)*(1-d[y].transpose()))
                    tempN=numpy.sum(tempN)
                    tempD=(1-0.5*(1-x)*(1-d[y].transpose()))
                    tempD=numpy.sum(tempD)
                    if tempD==0:
                        self.coOccurMatrix[ctrx][y] = self.coOccurMatrix[y][ctrx] = 0.0
                    else:
                        self.coOccurMatrix[ctrx][y]=self.coOccurMatrix[y][ctrx]=numpy.divide(tempN,tempD)
            ctrx+=1

        print 'co-occurance matrix is done for layer'

    # method to obtain centroid of the clusters and prepare the weight matrix for above layer
    def prepareCentroidMatrix(self, phsample, objCRBM):
        mean= None; phsampleNew=None; temp=None
        #removing duplicate entries from phsamples
        temp = [tuple(row) for row in phsample]
        phsampleNew=numpy.unique(temp)


        self.InterLayerWeightAbove= self.calculateCentroids(objCRBM, phsampleNew)
        self.numberOfCertroidIdentified= self.InterLayerWeightAbove.shape[0]


    def calculateCentroids(self, objCRBM,phsampleNew):
        centroids=[]
        cluster=[]
        data= numpy.asarray(self.inputTrainData)
        for x in phsampleNew:
            temp=[]
            for y in data:
                i, temp1= objCRBM.sample_h_given_v(y)
                if all((temp1== x)):
                    temp.append(y)
            if temp.__len__()!= 0:
                cluster.append(temp)
        cluster = numpy.asarray(cluster)
        for clus in cluster:
            count=0
            clst= numpy.asarray(clus)
            if len(clst)==0:
                continue
            lenght=clst.shape[1]
            tempx=numpy.zeros(lenght)
            for j in clst:
                tempx = numpy.add(tempx,j)
                count+=1
            tempx = numpy.divide(tempx,count)
            centroids.append(tempx)
        self.cluster=cluster
        return numpy.asarray(centroids)

     # method to caluculate weighted average,and weighted standard deviation
    def weighted_avg_and_std(self, values=None, weights=None):
        """
            Return the weighted average and standard deviation.

            values, weights -- Numpy ndarrays with the same shape.
        """

        wt = []
        for x in weights:
            if x != 0:
                wt.append(1.0)
            else:
                wt.append(0.0)
        average = numpy.average(values, weights=wt)
        variance = numpy.average((values - average) ** 2, weights=wt)  # Fast and numerically precise
        return (average, numpy.sqrt(variance))

    def sampleHgivenV(self, vSample=None):

        mean, hsample=self.crbm.sample_h_given_v(vSample)
        temp=[tuple(row) for row in hsample]
        hsampleunique= numpy.unique(temp)

        return [hsample, hsampleunique]

        print "method to propagate activation to above layer"

