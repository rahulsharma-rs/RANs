__author__ = 'Rahul'

from Layer import Layer
from scipy.spatial import distance
from mnist import *
from imgutil import *

class RAN(object):

    def __init__(self):

        self.modelID = None
        self.numberOfLayers = 0 # creates RAN with only one layer grows as the number of layers increases
        self.sFactor=0.5 #solipsismfactor
        self.rFactor=0.5#regulatory factor
        self.cCenter=[] #convergence point
        self.conceptIdentifier={}# dictionary of clustering algorithm "rbm" "meanshift" "sc" "k-mean" "afp" with parameter

        self.layer =[] #list holding the objects of convex concept layers, and grows with the new layer
        self.nCLayer=[] #list to hold objects of non-convex concept layers
        self.rawData = None # raw data imported from the csv file
        self.stdDev=None # standard deviation of data used when data un-whitening is done

        self.testData= None # data to test the result
        self.NumberOfNodes = [] # list holding the number of convex nodes in each convex layer
        self.NumberOfNonConvexNodes=[]#list  holding the number of non-convex nodes in each non-convex layer
        self.impactInfluencer=3# the power multiplier of the impact factor

        self.weight =[] # list of weight matrix at each layer to propagate upward
        self.weightDown=[]# list of weight matrix at each layer to propagate downward
        self.weightCoOccurance=[]# list of weight matrix for co-occurance of nodes in each layer
        self.nConvexWeightCoOccurance = []  # list of weight matrix for co-occurance of nodes in each non-convex layerlayer
        self.nConvexWeights=[] # list of non-convex weights learned for each layer
        self.totalInput=[] #list of total input received by the nodes in each layer
        self.nConvexTotalInput = []  # list of total input received by the nodes in each non-convex layer
        self.nodesActivation=[] # size changes based upon the layer of the node
        self.testNodesActivation= []


        self.maximumActivationCountForEachTopLayerNode=[] # for keeping count of maximum activation attained by each node on the top layer


        self.intermediateData = [] # new-data to train the newly created layer, it is obtained by transforming the rawData
        self.intermediateData1 = []                             # comprehendable to above layer
        self.intermediateTestData = []
        self.intermediateDataObtained= [] # storing the intermediate obtained while propagating downwards
        self.testData=[]
        self.dataProjection=[]# this list holds the activation of nodes while propagating downwards
        self.dproj=[]
        self.startPoint=[]
        self.actualActivation=[]
        self.error=[]
        self.error1=[]

        self.stoppingCriteria = None

        self.classLabels=None #  labels learned, index->value == concept_representative-> training_labels
        self.labels=None #sorted array of labels of test data in ascending order used for vlidation
        self.unique_labels=None #contains arrave of unique labels in test data
        self.label_frequency=None # contains the frequency of each label in test data
        self.labelCount=None# count of labels in the data set

        self.observation=None


    def createLayer(self,data1, objRAN, isconvex=True,layerId=None):
        if isconvex==True:
            print "Creating convex concept Layer: ", objRAN.numberOfLayers+1
            objRAN.numberOfLayers +=1 # incrementing the counter for number of layers
            object = Layer(data=data1, numberOfInputs=data1.shape[1],id=objRAN.numberOfLayers) # creating the layer object
            self.layer.append(object) # creating the new layer object and appending to the layer list
        if isconvex==False:
            print "Creating non-convex concept layer",objRAN.numberOfLayers
            object=Layer(data=data1,numberOfInputs=data1.shape[1],id=layerId,isconvex=False,objRAN=objRAN)# creating layer object for non-convex concept
            self.nCLayer.append(object)# appending the layer object of non-convex concept to the list


    def train_RAN_Upward(self, ran,isConvex=False,conceptIdentifier=None):
        currentLayer=currentNCLayer=currentCLayer=0
        self.conceptIdentifier=conceptIdentifier
        stoppingCriteria= 1 # self.layer[(len(self.layer)-1)].numberOfCertroidIdentified >1  stopping criteria checking the number of
                               # identified centroids in the highest layer, if it is 1 the stop.

        ran.createLayer(ran.rawData, ran) # creating the first layer
        #ran.NumberOfNodes.append(ran.rawData.shape[1])
        temp11, temp21, cooccur,totalInput= ran.layer[currentLayer].layerOperation(currentLayer+1,conceptIdentifier=self.conceptIdentifier)# performs intraLayer training for the first layer with the raw data
        self.updateParameters(temp11, temp21,cooccur,totalInput, ran)
        ran.nCLayer.append([None]) # appending None of non-convex layer at input level
        ran.nConvexWeights.append([None])#appending None to non-convex layer-weights at input level
        ran.NumberOfNonConvexNodes.append([None])#appending None to number of non-convex nodes  at input level
        ran.nConvexWeightCoOccurance.append([None])#appending None to non-convex coOccurance-weights at input level
        ran.nConvexTotalInput.append(None)#appending None to non-convex total input received by each node at input level

        while stoppingCriteria!=2:
            temp1 = None; temp2 = None; temp3 =[];temp4 =None; temp5= [];temp6=None

            if isConvex==True:
                temptx=[]
                for w in ran.weight[currentCLayer]:
                    data = self.intermediateData[currentCLayer]
                    temptx.append(self.propagateUpward(w, data))

                tempxyt=numpy.asarray(temptx).transpose()
                ran.intermediateData.append(tempxyt)


                # now pointing to the new layer
                currentLayer += 1
                currentCLayer += 1  ##pointing to current convex layer
                # creating convex-concept layer
                ran.createLayer(ran.intermediateData[currentLayer], ran)  # creating the new layer.
                temp1, temp2, cooccur, totalInput = ran.layer[currentCLayer].layerOperation(
                    currentCLayer + 1,conceptIdentifier=self.conceptIdentifier)  # performs intraLayer training for the new layer with the new data
                if temp2 == 1:
                    self.updateParameters(None, temp2, cooccur, totalInput, ran)
                else:
                    self.updateParameters(temp1, temp2, cooccur, totalInput, ran)
                if temp2 <= 1 or self.numberOfLayers >1:
                    stoppingCriteria = 2

            else:
            # Iterating through all the input of the lower layer, and create train data for the above layer
                temptx = []
                for w in ran.weight[currentCLayer]:
                    data = self.intermediateData[currentCLayer]
                    temptx.append(self.propagateUpward(w, data))

                tempxyt = numpy.asarray(temptx).transpose()
                ran.intermediateData.append(tempxyt)
                # now pointing to the new layer
                currentLayer +=1
                currentCLayer += 1  ##pointing to current convex layer
                #creating convex-concept layer
                ran.createLayer(ran.intermediateData[currentLayer], ran) # creating the new layer.
                temp1, temp2,cooccur,totalInput= ran.layer[currentCLayer].layerOperation(currentCLayer+1,conceptIdentifier=self.conceptIdentifier)# performs intraLayer training for the new layer with the new data
                if temp2==1:
                    self.updateParameters(None, temp2,cooccur,totalInput, ran)
                else:
                    self.updateParameters(temp1, temp2,cooccur,totalInput, ran)
                #creating non-convex concept layer
                currentNCLayer += 1#pointing to current non-convex layer
                ran.createLayer(ran.intermediateData[currentLayer], ran, isconvex=False, layerId=currentNCLayer)
                temp1, temp2, cooccur, totalInput = ran.layer[currentNCLayer].layerOperation(
                    currentNCLayer + 1,conceptIdentifier=self.conceptIdentifier)  # performs intraLayer training for the new non-convex layer with the new data
                #updating parameter for non-convex layer features
                if ran.nCLayer[currentNCLayer].numberOfNonConvexNodes==1:
                    ran.updateParameters(ran.nCLayer[currentNCLayer].nCWeight,
                                         ran.nCLayer[currentNCLayer].numberOfNonConvexNodes,
                                         ran.nCLayer[currentNCLayer].coOccurMatrix,
                                         ran.nCLayer[currentNCLayer].individualOccuranceCount, ran, isConvex=False)
                else:
                    ran.updateParameters(ran.nCLayer[currentNCLayer].nCWeight, ran.nCLayer[currentNCLayer].numberOfNonConvexNodes,
                                     ran.nCLayer[currentNCLayer].coOccurMatrix,
                                     ran.nCLayer[currentNCLayer].individualOccuranceCount, ran, isConvex=False)
                #
                for x in ran.intermediateData[currentLayer]:
                    temp5.append(self.propagateUpward(ran.nConvexWeights[currentNCLayer], x, isConvex=False))
                temp6 = numpy.asarray(temp5)
                ran.intermediateData.append(temp6)
                currentLayer += 1
                if temp2 <= 1 or self.numberOfLayers>1:
                    stoppingCriteria = 2

    def propagateUp(self, ran,isConvex=False):
        currentLayer = currentNCLayer = currentCLayer = 0
        stoppingCriteria = 1
        while stoppingCriteria != 2:
            temp5 = []
            # Iterating through all the input of the lower layer, and create train data for the above layer
            if isConvex==True:
                temptx = []
                for w in ran.weight[currentCLayer]:
                    data = self.intermediateData1[currentCLayer]
                    temptx.append(self.propagateUpward(w, data))
                tempxyt = numpy.asarray(temptx).transpose()
                ran.intermediateData1.append(tempxyt)

                # now pointing to the new layer
                currentLayer += 1

                currentNCLayer += 1  # pointing to current non-convex layer
                currentCLayer += 1  ##pointing to current convex layer
                if currentLayer >= ran.weight.__len__() - 1:
                    stoppingCriteria = 2
            else:
                temptx = []
                for w in ran.weight[currentCLayer]:
                    data = self.intermediateData1[currentCLayer]
                    temptx.append(self.propagateUpward(w, data))
                tempxyt = numpy.asarray(temptx).transpose()
                ran.intermediateData1.append(tempxyt)

                 # now pointing to the new layer
                currentLayer += 1

                currentNCLayer += 1  # pointing to current non-convex layer
                currentCLayer += 1  ##pointing to current convex layer

                for x in ran.intermediateData1[currentLayer]:
                    temp5.append(self.propagateUpward(ran.nConvexWeights[currentNCLayer], x,isConvex=False))
                temp6 = numpy.asarray(temp5)
                ran.intermediateData1.append(temp6)

                #ran.createLayer(ran.intermediateData1[currentLayer], ran)  # creating the new layer.
                #temp1, temp2, cooccur, totalInput = ran.layer[currentLayer].layerOperation(currentLayer + 1)  # performs intraLayer training for the new layer with the new data
                if currentLayer >= ran.weight.__len__()-1:
                    stoppingCriteria = 2
                    # if dimension self.intermediateData id 1 that is only one column then come out of while loop
                    # else create a new layer with data = self.intermediateData and iterate in the while loop satisfying the while condition

                    # can perform the operation in the last layer using self.intermediate data

    def updateParameters(self,weightMatrix,nodeCount,coOccur,totalInput, objRAN,isConvex=True):
        if isConvex==True:#updating parameters for convex concept layer
            if weightMatrix!=None:
                objRAN.weight.append(weightMatrix)# appending the new weight matrix to the list
                objRAN.totalInput.append(totalInput)# appending the total input for each node at each layer.
            objRAN.weightCoOccurance.append(coOccur)# appending the new courrance matrix
            objRAN.NumberOfNodes.append(nodeCount)# appending the new count of visible nodes to the list
        if isConvex==False:#updating parameters for non-convex concept layer
            objRAN.nConvexWeights.append(weightMatrix)
            objRAN.NumberOfNonConvexNodes.append(nodeCount)
            objRAN.nConvexWeightCoOccurance.append(coOccur)
            objRAN.nConvexTotalInput.append(totalInput)

   # propagate the activation upward using the weight matrix and activations of the lower layers nodes
    def propagateUpward(self, weight, activation,isConvex=True):
        temp=[];normalizer= None; temp1=None
        if isConvex==True:
            #matrix operation logic for eucledian distance calculation
            yxx = numpy.subtract(activation, weight)#
            yxx1 = numpy.square(yxx)
            yxx2 = numpy.sum(yxx1, axis=1)
            yxx3 = numpy.divide(yxx2.astype(float), float(weight.size))
            return self.similarityConverter3(yxx3)

        if isConvex==False:
            #matrix multiplication logic
            temp = numpy.prod(1-numpy.multiply(weight, activation.transpose()),axis=1)

            return (1-temp)
            #return self.similarityConverter2(temp)

    def propagateUpwardOld(self, weight, activation,isConvex=True):
        temp = [];
        normalizer = None;
        temp1 = None
        for w in weight:
            temp.append(distance.sqeuclidean(w, activation))  # calculating the eucladian diatance between the points
        # normalizer=numpy.sqrt(len(activation))
        normalizer = len(activation)
        temp = numpy.asarray(temp)  # converting to array
        temp1 = numpy.divide(temp, normalizer)

        # temp1= numpy.sqrt(temp1) # it hardly make any difference

        return self.similarityConverter3(temp1)

    #this method converts the distance measures into the similarity
    def similarityConverter(self,activation):
        x= -(activation**2)
        y=(((numpy.exp(x)-1/numpy.exp(1)))/(1-(1/numpy.exp(1))))**(numpy.exp(1))**2
        return y

    def similarityConverter1(self,activation):
        #x= -(activation**2)
        #y=(((numpy.exp(x)-1/numpy.exp(1)))/(1-(1/numpy.exp(1))))**(numpy.exp(1))**2
        return numpy.exp(-((numpy.exp(1)-numpy.log(1-abs(activation)))**2-numpy.exp(1)**2)**2)

    def similarityConverter2(self,activation):
        temp=numpy.square(1-(numpy.sqrt(activation)))
        return temp

    def similarityConverter3(self,activation):
        temp=numpy.square(1-(activation**(float(1)/3)))
        return temp

    def similarityConverter4(self,activation):
        temp=numpy.power(activation,4.0)-2*numpy.power(activation,3.0)+numpy.power(activation,2.0)
        return temp
    def errorGradient(self,K=None,Ax=None,w=None):
        nu= 4*K(w-Ax)[(K*(w-Ax)**2)**(1/3)-1]
        du= 3*(K*((w-Ax)**2))**(2/3)

    def errorCalculation(self,weight,activation,AA=None,EA=None):
        temp=[];normalizer= None; temp1=None;error=[];
        for w in weight:
            temp.append(distance.sqeuclidean(w,activation))# calculating the eucladian diatance between the points
        x=y=0
        weight=numpy.transpose(weight)
        temp=numpy.asarray(temp) # converting to array
        n=len(activation)
        for i in range(0,len(activation)):

            err=[]
            acIndex=0
            y=0
            for ax in range(0,len(temp)):
                k=temp[y]/(n*(weight[x][y]-activation[x])**2)
                deltaError= -(EA[x]-AA[x])
                #deltaOut= (1-(1/numpy.sqrt(temp[y])))
                #deltaOut=(2*(temp[y]**float(1)/3)-1)/(3*temp[y]**float(2)/3)
                #deltaSimilarity= (2*activation[x]-2*weight[x][y])*k
                K1=4*k*(weight[x][y]-activation[x])
                opN1=1-numpy.power(k*numpy.square(weight[x][y]-activation[x]),float(1)/3)
                opD1=numpy.power(k*numpy.square(weight[x][y]-activation[x]),float(2)/3)
                dS=K1*(opN1)/3*(opD1)
                #err.append(deltaError*deltaOut*deltaSimilarity)
                err.append(deltaError*dS)
                y+=1;acIndex+=1
            err=numpy.asarray(err)
            error.append(err)
            x+=1
        return error


        #temp1=numpy.divide(temp,normalizer)

        #temp1= numpy.sqrt(temp1) # it hardly make any difference

        return self.similarityConverter2(temp1)
    def train_RAN_Downward(self, objRAN,data=None, learningRate=0.9, momentumFactor=0.1):
        intermediateDataObtained=data
        for i in range(1,0, -1):
            tempWeight=objRAN.weightDown[i-1]
            temp=[]
            for act in intermediateDataObtained[i]:
                temp.append(objRAN.propagateDownward(weight=tempWeight,activation=act,competition=0, layerIndex=i))
            temp= numpy.squeeze(numpy.asarray(temp))# getting rid of unwanted dimension of the array
            objRAN.intermediateData1.append(temp)
            intermediateDataObtained[i-1]=temp
        objRAN.intermediateDataObtained=intermediateDataObtained


    def propagateDownward(self,weight, activation, competition=1, layerIndex=None):
        temp=[];ind=[];a=[]
        #if there is the competition among the nodes the the highest, activation node wins the competition
        #if competition=1 then retain the highest activation
        for x in range(0,len(activation)):
            #making activation of all the nodes to zero except the node having highest activation
            if activation[x]!= activation.max():
                a.append(0)
            else:
                a.append(1)
                ind.append(x)
        a=numpy.asarray(a)
        if competition==1:
            temp.append(self.calcActivation(weight,a))# calculating the eucladian diatance between the points
        else:
            temp.append(self.calcActivation(weight,activation))
        temp1=self.calcActivation1(weight,activation,index=ind,lamda=01,layerInfo=layerIndex)
        return temp1

    #method to calclate activation by backpropagation
    def calcActivation1(self,weight,activation,index=0,lamda=0.01,layerInfo=None):
        expectedActivation=activation;w=weight;stopCriteria=1;temProj=[];temDevi=[];temperror=[]
        oldDiff=0;ctr=0
        wt=numpy.transpose(w);diff=1
        #tempAct=numpy.zeros(wt.shape[1])
        #tempAct=numpy.random.rand(wt.shape[1])
        tempAct=self.startPoint[layerInfo-1]
        for val in index:
         self.cCenter.append(wt[val])
        #tempAct=wt[index]# assigning the weight as the activation of highest probable node
        t1= numpy.transpose(numpy.transpose(numpy.transpose(wt)*expectedActivation))
        temProj.append(tempAct)
        prevAct=tempAct
        actualActivation=self.propagateUpwardOld(wt,tempAct)
        #regulate the activation
        t2=[]
        actualActivation=(1-self.rFactor)*actualActivation + (self.rFactor)*self.regulate(self.weightCoOccurance[layerInfo],actualActivation)
        if actualActivation.min() < 0 or actualActivation.max() > 1:
            for tx2 in actualActivation:
                if tx2 > 1:
                    t2.append(1.0)
                elif tx2 < 0:
                    t2.append(0.0)
                else:
                    t2.append(tx2)
            actualActivation = numpy.asarray(t2)
        error=expectedActivation-actualActivation
        temperror.append(error)
        #error= self.errorCalculation(wt,tempAct,AA=actualActivation,EA=expectedActivation)
        '''
        t1=[]
        error1=numpy.transpose(error1)
        for v1 in (error1):
            t1.append(numpy.sum(v1))
        t1=numpy.asarray(t1)
        '''
        eImp= numpy.transpose(numpy.transpose(numpy.transpose(numpy.transpose(numpy.asarray(error)))*expectedActivation))  # importance of error based upon the expectation
        #intAct=self.lateralReg(layerIndex=layerInfo,activation=actualActivation)
        #actualActivation= self.sFactor*actualActivation + (1-self.sFactor)*intAct
        while stopCriteria!=2:
            ind=0
            temp1=[]


            for x in (tempAct):
                y=weight[ind]
                temp=0
                temp=(((y-x))*error)
                temptest=numpy.sum(temp)
                temp2=lamda*temptest
                temp1.append(temp2)
                ind+=1
            temp1=numpy.asarray(temp1)/weight.shape[1]

            ####ensuring the values between 0 and 1
            temptact=[]
            ctr1=0
            for delctr in temp1:
                if delctr>0:
                    #for adding positive activation
                    #zero addition when tempAct =1 and max addition when tempAct=1
                    act=tempAct[ctr1]+temp1[ctr1]*(1-tempAct[ctr1])
                    if act>1:# if the new activation exceedes the limits the do nothing
                        temptact.append(1)
                    elif act<0:
                        temptact.append(0)
                    else:
                        temptact.append(act)
                else:
                    #for adding necative activation
                    #no reduction when tempAct =0 and max reduction when tempAct=1
                    act=tempAct[ctr1]+temp1[ctr1]*(tempAct[ctr1])
                    if act>1:# if the new activation exceedes the limits the do nothing
                        temptact.append(1)
                    elif act<0:
                        temptact.append(0)
                    else:
                        temptact.append(act)
                ctr1+=1
            tempAct=numpy.asarray(temptact)
            tempAct=tempAct
            #regulation layer L-1
            t1=[]
            tempAct=(1-self.rFactor)*(tempAct) + (self.rFactor)*self.regulate(self.weightCoOccurance[layerInfo-1],(tempAct))
            if tempAct.min()<0 or tempAct.max()>1:# checking if activation is within bounds
                for tx1 in tempAct:
                    if tx1>1:
                        t1.append(1.0)
                    elif tx1<0:
                        t1.append(0.0)
                    else:
                        t1.append(tx1)
                tempAct=numpy.asarray(t1)
                newAct = tempAct
            else:
                newAct=tempAct
            actualActivation=self.propagateUpwardOld(wt,tempAct)
            #regulation at layer L

            t2=[]
            actualActivation=(1-self.rFactor)*actualActivation + (self.rFactor)*self.regulate(self.weightCoOccurance[layerInfo],actualActivation)
            if actualActivation.min()<0 or actualActivation.max()>1:
                for tx2 in actualActivation:
                    if tx2>1:
                        t2.append(1.0)
                    elif tx2<0:
                        t2.append(0.0)
                    else:
                        t2.append(tx2)
                actualActivation=numpy.asarray(t2)

            #intAct=self.lateralReg(layerIndex=layerInfo,activation=actualActivation)
            #actualActivation= self.sFactor*actualActivation + (1-self.sFactor)*intAct
            temProj.append(newAct)
            temDevi.append(numpy.divide(distance.sqeuclidean(newAct,prevAct),len(newAct)))
            prevAct=tempAct

            #logic for window operation
            if temDevi.__len__()>100:
                x=numpy.asarray(temDevi[temDevi.__len__()-50:])
                diff= numpy.std(x)
                if oldDiff-diff!=0:
                    oldDiff=diff
                else :
                    diff =0.0
            if ctr<5000:#ctr<790:
            #if diff>0.001 or diff==None:
                stopCriteria=1
                error=expectedActivation-actualActivation
                temperror.append(error)
                #error= self.errorCalculation(wt,tempAct,AA=actualActivation,EA=expectedActivation)
                eImp= numpy.transpose(numpy.transpose(numpy.transpose(numpy.transpose(numpy.asarray(error)))*expectedActivation))
                ctr+=1
            else:
                stopCriteria=2
        tempProj=numpy.asarray(temProj)
        temperror=numpy.asarray(temperror)
        self.actualActivation[layerInfo].append(actualActivation)
        self.error.append(error)
        #self.error1[layerInfo].append(temperror)
        self.dataProjection[layerInfo-1]=tempProj
        return tempAct

    #regulation with lateral correlation
    def regulate(self,weightMatrix=None,activation=None):
        #normalizing to the relevance matrix based upon the effectiveness of the weight over the activation
        newAct=[]
        #si =(2*abs(weightMatrix-0.5))**self.impactInfluencer
        si = (2 * (weightMatrix - 0.5)) ** self.impactInfluencer
        for x in range(0,len(activation)):
            temp1=sum(si[x]*((activation[x]*weightMatrix[x])+(1-activation[x])*(1-weightMatrix[x])))
            temp2=si[x][x]*((activation[x]*weightMatrix[x][x])+(1-activation[x])*(1-weightMatrix[x][x]))
            newAct.append(((temp1-temp2)/(sum(si[x])-si[x][x])))
        newAct=numpy.asarray(newAct)
        return newAct

    #lateral regulation
    def lateralReg(self,layerIndex=None, activation=None):
        tempAct=[];intAct=None
        PH=PNH=PtEiH=PtEiNH=1
        index=0
        pAct=PEi=self.totalInput[layerIndex]/self.rawData.shape[0]#array of probability of activation of each node
        pNoAct=PNEi= 1-pAct# probabilitiy array when the node is not active


        for x in self.weightCoOccurance[layerIndex]:

            PH=self.totalInput[layerIndex][index]/self.rawData.shape[0]#P(H)
            PNH=1-PH #P(-H)
            try:
                PHEi=x/pAct #P(H|Ei)
            except ZeroDivisionError:
                PHEi=0

            try:
                PHNEi=(self.totalInput[layerIndex][index]-x)/(self.rawData.shape[0]-pAct) #P(H|-Ei)
            except ZeroDivisionError:
                PHNEi=0
            PtEi=activation #Pt(Ei)= Act(Ei)
            PtNEi=1-PtEi #Pt(-Ei)

            PtHEi= (PHEi*PtEi)+(PHNEi*PtNEi) #Pt(H|Ei)

            PtNHEi=1-PtHEi #Pt(-H|Ei)
            try:
                PtEiNH=(PtNHEi * PNEi)/PH #Pt(Ei|-H)
            except ZeroDivisionError:
                PtEiNH=0
            try:
                PtEiH=(PtHEi*PEi)/PNH #Pt(Ei|H)
            except ZeroDivisionError:
                PtEiH=0

            try:
                intAct= (PH * PtEiH.prod())/((PH * PtEiH.prod())+(PNH * PtEiNH.prod()))
            except ZeroDivisionError:
                intAct=0

            tempAct.append(intAct)
            index+=1
        tempAct=numpy.asarray(tempAct)

        return tempAct

    #method to handle array of sigmoid
    def arySigmoid(self,arr):
        temp=[]
        for x in arr:
            temp.append(self.sigmoid(x))
        temp=numpy.asarray(temp)
        return temp

    def calcActivation(self, weight, activation):

        #activations w.r.t each activation, each row in the 2d array represents the activation
        temp=numpy.transpose(weight*activation.transpose())
        temp1=temp[0]
        temp=numpy.transpose(temp)
        count=0
        tem=[]
        #adding the rows
        for x in range(0,temp.shape[0]):
            tem.append(numpy.sum(temp[x]))
            count +=1
        #calculating the average of all the rows
        tem=numpy.asarray(tem)
        temp1=tem
        return temp1



    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+numpy.exp(-z))

    def transformActivation(self, a):
        temp=[]
        for x in a:
            temp.append(numpy.exp(x-1)*(3*x**2-2*x**3))
        temp= numpy.asarray(temp)
        return temp


    def weightedAverage(self, W, A):
        return [numpy.dot(numpy.transpose(A),W)/numpy.sum(A)]

    def test(self,  objRAN):
        for currentLayer in range(0, objRAN.numberOfLayers-1):
            temp3 =[]
            # Iterating through all the input of the lower layer, and create train data for the above layer
            for x in objRAN.intermediateTestData[currentLayer]:
                temp3.append(self.propagateUpward(objRAN.weight[currentLayer], x))
            temp4 = numpy.asarray(temp3)
            objRAN.intermediateTestData.append(temp4)
            print "x"

    def activationCount(self, objRAN):
        count =0
        hsample, hsampleunique= objRAN.layer[objRAN.numberOfLayers-4].sampleHgivenV(objRAN.intermediateData[objRAN.numberOfLayers-4])
        for i in range(0,hsampleunique.shape[0]):

            objRAN.maximumActivationCountForEachTopLayerNode.append(0)

        for i in hsampleunique:
            for j in hsample:
                if (i==j).all():
                    objRAN.maximumActivationCountForEachTopLayerNode[count]+=1

            count +=1
    #function to normalize an array of data between [0 1]
    def normalizeData(self,data):
        num=data-data.min()
        den=data.max()-data.min()
        normData=None
        if den!=0:
            normData=num/den
        else:
            normData=data
        return normData

    #derivative of sigmoid function in terms of output
    def dsigmoid(self,y):
        return y*1.0 - y**2

