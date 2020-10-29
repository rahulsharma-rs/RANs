from UtilsRAN import processInputCSVFile,labelNodes,plot_confusion_matrix, whiten,plotROC
import RAN_kfold as RAN
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedShuffleSplit,ShuffleSplit
import sklearn
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import  Pipeline
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,accuracy_score
from Processing_split_data import processData
import numpy
import matplotlib.pyplot as plt

import matplotlib as mp
import pandas
from pandas.plotting import parallel_coordinates

def main():



    #new chosesn datasets
    #path="/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/adriano/adriano-data.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/Data_cortex_Nuclear/mice_with_class_label.csv"
    path="/Users/mg/Dropbox/MyWork/RAN/data/iris_with_label.csv"
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/GlassIdentificationDatabase/RANsform.csv'#identify k=6 abstract concepts
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/WineRecognitionData/RansForm.csv'#identify the categories k=3
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/breastCancerDatabases/699RansForm.csv'#identify categories k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/breastCancerDatabases/569RansForm.csv'# identifies categories k=2
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/UCI_HAR_Dataset.csv"
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/nonConvexToyData2.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/MammographicMassData/RansForm1.csv' #identify Category k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/HTRU/RansForm.csv' #identify category k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/DrugConsumptionData/RansForm_Alcohol.csv' #check for this data there are some results wit 75%accuracy
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/CreditApproval/RansForm.csv' # identify category with k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/CreditApproval/german.csv' #german credit data k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/bank/RansForm.csv' #identify categories with k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/DrugConsumptionData/RansForm_Ketamine.csv' #k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/DrugConsumptionData/RansForm_METH.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/DrugConsumptionData/RansForm_vsa.csv'

    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleep-With-Artificial_label-RAN.csv' #olddata
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleep-With-Artificial_multiple_label-RAN.csv'#olddata
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleepgenericstudata/sleep-data-RAN-8-Labels.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleepgenericstudata/sleep-data-RAN-2-Labels.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleepgenericstudata/sleep-data-RAN-7-Labels.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/toydata5clustersRAN.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/toydataNew5clustersRAN.csv'
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/toydata9clustersRANWH.csv'#
    #path='/home/rahul/Dropbox/MyWork/RAN/adrianoProj/RansForm.csv'



    #path="/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/mnist1000data.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/nswl.csv"

    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/UCI_HAR_Dataset_whitened.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/MovementAALdata.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/Occupancydatatest.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/activityRecognitionSingleChestMountedSensor.csv"
    #path = "/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/epilemticCisure.csv"

    rawData,header = processInputCSVFile(path,normalize=True)
    numpy.random.shuffle(rawData)#shuffeling the input data if it is in an order
    '''
    #Logic To Whiten the input data
    xx=rawData.T
    data,stdDev=whiten(xx[:xx.__len__()-1].T,check_finite=True)
    #x_data=sklearn.preprocessing.normalize(data)
    header=numpy.asarray([data.max()]*10+[100])
    data = data.T
    labels=rawData.T[rawData.T.__len__()-1].T
    data= numpy.asarray(data.tolist()+[labels.tolist()])
    data=data.T*(1/header)
    rawData=data
    '''
    labels = rawData.T[rawData.T.__len__() - 1].T
    label_count=numpy.unique(labels).size
    unique_labels=numpy.unique(labels)
    report1 = []
    report_AUC=[]
    avPrecision=[[], [], [], [], [], []];avRecall=[[], [], [], [], [], []];avF1=[[], [], [], [], [], []];avAccuracy=[[], [], [], [], [], []]
    listPrecision = [[], [], [], [], [], []];listRecall = [[], [], [], [], [], []];listF1 = [[], [], [], [], [], []];listAccuracy = [[], [], [], [], [], []]
    #loop for number of iterations of the experiment
    for topiter in range(0, 1): #loop 30 times
        iter1 = 0

        tr_size = 0.9  # size of train data
        #loop for the 9 Research Designs of the experiment
        for iter1 in range(1,2):
            r = []
            reports = []
            acc = [[], [], [], [], [], []]
            pr = [[], [], [], [], [], []]
            rec = [[], [], [], [], [], []]
            fone = [[], [], [], [], [], []]
            sup = [[], [], [], [], [], []]
            foneWeight= [[], [], [], [], [], []]
            lstPrecision = [[], [], [], [], [], []];lstRecall = [[], [], [], [], [], []];lstF1 = [[], [], [], [], [], []];lstAccuracy = [[], [], [], [], [], []]
            modelName = []
            pltctr = 0
            fprCurve = []
            tprCurve = []
            fprCurve1 = []
            tprCurve1 = []
            prCurve = []
            rCurve = []
            thresCurve = []
            classLabels = []
            AUC=[]
            modelArc = []
            iter = 0
            if tr_size==0.1:
                print ("x")
            #rawData=rawData.T[:rawData.T.__len__() - 1].T
            #data_train, data_test, labels_train, labels_test = train_test_split(rawData, labels, test_size=0.20, random_state=42)
            kf1=StratifiedKFold(n_splits=10)
            kf2=StratifiedShuffleSplit(n_splits=10,train_size=tr_size,test_size=1-tr_size,random_state=0)
            tr_size=tr_size-0.1
            kf = KFold(n_splits=10)
            kf3=ShuffleSplit(n_splits=10,test_size=0.4,train_size=0.6,random_state=0)
            isConvex=True
            #for tr1,ts1 in kf1.split(rawData,labels):
               # print tr1
                #print '/n'
               # print ts1
            #loop for 10-fold stratified cross validation
            for train, test in kf2.split(rawData, labels):
            #for train, test in kf.split(rawData):
                ran=None
                ran = RAN.RAN()
                ran1=RAN.RAN()
                X_train, X_test, Y_train, Y_test = rawData[train], rawData[test], labels[train], labels[test]
                trainD,sortedData,label,unique_label, labelFrequency=processData(data=X_train)# processing training data (here the input data is seperated by labels)
                ran.labels=label
                ran.unique_labels=unique_label
                ran.label_frequency=labelFrequency
                ran.labelCount=numpy.unique(labels).size
                Y_train_temp= numpy.zeros(Y_train.size)
                Y_test_temp= numpy.zeros(Y_test.size)
                ctrxi=0
                '''
                #----------------------------------------------------------------------------------------
                #preparing traing and test label according to 2 astract concepts in in UCIHAR data
                #scalar=StandardScaler()
                #X_train_temp= scalar.fit(rawData[train])
                #X_test_temp=scalar.fit(rawData[test])
                
                for xi in Y_train:
                    if xi in ran.unique_labels[:3]:
                        Y_train_temp[ctrxi]=0
                    else:
                        Y_train_temp[ctrxi]=1
                    ctrxi+=1
    
                ctrxi = 0
                for xi in Y_test:
                    if xi in ran.unique_labels[:3]:
                        Y_test_temp[ctrxi] = 0
                    else:
                        Y_test_temp[ctrxi] = 1
                    ctrxi += 1
                #-----------------------------------------------------------------------------------------------

                '''
                '''
                #preparing train test labels according to abstract concepts in toydata9clusters
                for xi in Y_train:
                    if xi in ran.unique_labels[:3]:
                        Y_train_temp[ctrxi] = 0
                    elif xi in ran.unique_labels[3:6]:
                        Y_train_temp[ctrxi]=1
                    else:
                        Y_train_temp[ctrxi] = 2
                    ctrxi += 1

                ctrxi = 0
                for xi in Y_test:
                    if xi in ran.unique_labels[:3]:
                        Y_test_temp[ctrxi] = 0
                    elif xi in ran.unique_labels[3:6]:
                        Y_test_temp[ctrxi]=1
                    else:
                        Y_test_temp[ctrxi] = 2
                    ctrxi += 1
               
                #-----------------------------------------------------------------------------------------------
                '''
                '''
                    # preparing traing and test label according to 2 astract concepts in in glass identification data 
                    # scalar=StandardScaler()
                    # X_train_temp= scalar.fit(rawData[train])
                    # X_test_temp=scalar.fit(rawData[test])
                
                for xi in Y_train:
                    if xi in ran.unique_labels[:3]:
                        Y_train_temp[ctrxi] = 0
                    else:
                        Y_train_temp[ctrxi] = 1
                    ctrxi += 1

                ctrxi = 0
                for xi in Y_test:
                    if xi in ran.unique_labels[:3]:
                        Y_test_temp[ctrxi] = 0
                    else:
                        Y_test_temp[ctrxi] = 1
                    ctrxi += 1
                # -----------------------------------------------------------------------------------------------
                 '''
                #---------------------------------------------------------------------------------------------
                ## logic to normalize labels to integer in general should be commented when experimenting
                #  with Glassidneitfication, and UCIHAR data

                Y_train_temp = (Y_train * 3).astype(int)
                Y_test_temp = (Y_test * 3).astype(int)

                reports.append("----------------------------------Pass %s---------------------------------------------------------------" % iter)

                # -------MLP training
                reports.append("-------------------Multi-layer perceptron model-----------------------")
                modelName.append("MLP")
                print ("-------------------------MLP----------------------------------")
                mlp = MLPClassifier( random_state=1,
                                        hidden_layer_sizes=(15), max_iter=700)
                mlp.fit(X_train, Y_train_temp)
                predictionMlp = mlp.predict(X_test)
                print(confusion_matrix(Y_test_temp, predictionMlp))
                print(classification_report(Y_test_temp, predictionMlp,digits=7))

                print ("accuracy: %s" % (accuracy_score(Y_test_temp, predictionMlp)))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, predictionMlp)))
                reports.append(classification_report(Y_test_temp, predictionMlp,digits=7))
                reports.append("---------------------------------------------------------------------------\n")
                acc[0].append(accuracy_score(Y_test_temp, predictionMlp))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, predictionMlp)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys = yfone
                pr[0].append(numpy.average(yp, weights=ys))
                rec[0].append(numpy.average(yr, weights=ys))
                fone[0].append(numpy.average(yf1, weights=ys))
                sup[0].append(numpy.sum(ys))
                        # ------------
                #-------------------------------------------------------------
                # logistic regression
                reports.append("----------------------Logistic Regression-------------------------------")
                print ("-------------------------Logistic Regression----------------------------------")

                modelName.append("Logistic Regresion")
                logreg=LogisticRegression(multi_class='multinomial',solver='newton-cg',max_iter=5,class_weight='balanced',C=1)
                mLogreg=logreg.fit(X_train,Y_train_temp)
                pred=mLogreg.predict(X_test)
                print(confusion_matrix(Y_test_temp, pred))
                print(classification_report(Y_test_temp, pred,digits=7))

                print ("accuracy: %s" % (accuracy_score(Y_test_temp, pred)))

                reports.append("Accuracy: %s"  % (accuracy_score(Y_test_temp, pred)))
                reports.append(classification_report(Y_test_temp, pred,digits=7))
                acc[1].append(accuracy_score(Y_test_temp, pred))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, pred)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys = yfone
                pr[1].append(numpy.average(yp, weights=ys))
                rec[1].append(numpy.average(yr, weights=ys))
                fone[1].append(numpy.average(yf1, weights=ys))

                sup[1].append(numpy.sum(ys))
                reports.append("--------------------------------------------------------------------------------")
                #-----------------------------------------
                #-------------------------------------------
                ### bernauli rbm
                print ("------------------------- rbm----------------------------------")
                reports.append("-------------------- RBM-----------------------------")
                modelName.append("RBM")
                rbm=BernoulliRBM()
                #logistic = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10,class_weight='balanced', C=1)
                logistic=LogisticRegression(multi_class='multinomial',solver='newton-cg',C=1)
                classifire=Pipeline(steps=[('rbm',rbm),('logistic',logistic)])
                rbm.learning_rate=0.001
                rbm.n_iter=500
                rbm.n_components=20
                #logistic.C=6000.0

                #with multiple classes
                #logistic.fit(X_train, (Y_train * 6).astype(int))
                #classifire.fit(X_train,(Y_train*6).astype(int))

                #with two classes
                logistic.fit(X_train, Y_train_temp)
                classifire.fit(X_train,Y_train_temp)

                prd=classifire.predict(X_test)

                # with multiple classes
                #print(confusion_matrix((Y_test * 6).astype(int), prd))
                #print(classification_report((Y_test * 6).astype(int), prd))

                # with two classes
                print(confusion_matrix(Y_test_temp, prd))
                print(classification_report(Y_test_temp, prd,digits=7))

                print ("accuracy: %s" % (accuracy_score(Y_test_temp, prd)))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prd)))
                reports.append(classification_report(Y_test_temp, prd,digits=7))
                reports.append("-------------------------------------------------------------------------------------")
                acc[2].append(accuracy_score(Y_test_temp, prd))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prd)
                #logic for building binary weight based upon f1 score
                yfone=[]
                for f1i in yf1:
                    if f1i==0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys= yfone
                pr[2].append(numpy.average(yp, weights=ys))
                rec[2].append(numpy.average(yr, weights=ys))
                fone[2].append(numpy.average(yf1, weights=ys))

                sup[2].append(numpy.sum(ys))
                #----------------------------------------------------------------------
                
                #--------------------------------------------------------------------------
                #K-nearest neighbor
                print ("-------------------------K-Neareast Neoghbor----------------------------------")
                #reports.append("--------------------------K-Neareast Neoghbor-------------------------")
                modelName.append("K-NN")
                clf=neighbors.KNeighborsClassifier(n_neighbors=30)#,weights='distance',leaf_size=5,algorithm='auto',)
                clf.fit(X_train,Y_train_temp)
                prdt=clf.predict(X_test)
                print(confusion_matrix(Y_test_temp, prdt))
                print(classification_report(Y_test_temp, prdt,digits=7))

                print ("accuracy: %s" % (accuracy_score(Y_test_temp, prdt)))
                #reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prdt)))
                #reports.append(classification_report(Y_test_temp, prdt,digits=7))
                #reports.append("----------------------------------------------------------------------------------------------")
                acc[3].append(accuracy_score(Y_test_temp, prdt))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prdt)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys =  yfone
                pr[3].append(numpy.average(yp, weights=ys))
                rec[3].append(numpy.average(yr, weights=ys))
                fone[3].append(numpy.average(yf1, weights=ys))

                sup[3].append(numpy.sum(ys))

                #-----------------------------------------------------------------------------

                # --------------------------------------------------------------------------
                # SGD classifier
                print ("-------------------------SGD Classifier----------------------------------")
                reports.append("-------------------------SGD Classifier---------------------------")
                modelName.append("SGD")
                sgd = SGDClassifier(alpha=0.0001,max_iter=5,epsilon=0.25,)
                sgd.fit(X_train, Y_train_temp)
                prdtt = sgd.predict(X_test)
                print(confusion_matrix(Y_test_temp, prdtt))
                print(classification_report(Y_test_temp, prdtt,digits=7))

                print ("accuracy: %s" % (accuracy_score(Y_test_temp, prdtt)))
                reports.append("accuracy: %s" % (accuracy_score(Y_test_temp, prdtt)))
                reports.append(classification_report(Y_test_temp, prdtt,digits=7))
                reports.append("-------------------------------------------------------------------------------------------------------------")
                acc[4].append(accuracy_score(Y_test_temp, prdtt))
                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(Y_test_temp, prdtt)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys =  yfone
                pr[4].append(numpy.average(yp, weights=ys))
                rec[4].append(numpy.average(yr, weights=ys))
                fone[4].append(numpy.average(yf1, weights=ys))

                sup[4].append(numpy.sum(ys))

                # -----------------------------------------------------------------------------

                # --------------------------------------------------------------------------------------------------------
                modelName.append("RAN")
                print ("-------------------------Regulated Activation Networks----------------------------------")
                reports.append("----------------Regulated Activation Networks--------------------------")
                ran.rawData=sortedData#trainD #initializing RAN with sorted training data
                #ran.rawData=sklearn.preprocessing.normalize(sortedData/stdDev) # trainD #initializing RAN with sorted training data and  whitening the train data
                ran.intermediateData=[]# initializing the intermediated data
                ran.intermediateData.append(ran.rawData)
                # training
                ran.train_RAN_Upward(ran,isConvex=isConvex,conceptIdentifier={"name":"k-mean","k":3})
                if isConvex==False and ran.intermediateData[ran.intermediateData.__len__()-1].shape[1]==1:
                    reports.append('Skipping RANs %f-fold validation as model identifies only one abstract concept ' %iter)
                    iter+=1
                    continue

                # ran.intermediateData1= ran.intermediateData
                ran.dataProjection = [None] * (ran.numberOfLayers)
                ran.intermediateTestData = [None] * (ran.numberOfLayers)
                # temp=[]
                ran.actualActivation = [[]] * (ran.numberOfLayers)
                for dt in ran.intermediateData:
                    ran.startPoint.append(numpy.random.rand(dt.shape[1]))

                tempw = []
                if ran.weight[0].min() < 0 or ran.weight[0].max() > 1:
                    # ran.weight[0]= 2*ran.sigmoid(ran.weight[0])-1
                    for x in ran.weight[0]:
                        if x.min() < 0 or x.max() > 1:
                            tempw.append(ran.normalizeData(x))
                        else:
                            tempw.append(x)
                    ran.weight[0] = numpy.asarray(tempw)
                    # ran.weight[0]=ran.normalizeData(ran.weight[0])
                # --------------------------------------------------------------------------------------------------------
                for w in ran.weight:
                    ran.weightDown.append(numpy.transpose(w))

                # --------------------------------------------------------------------------------------------------------

                #----------------------------------------------------------------------------------------------------------\
                '''
                #logic for several active-subject analysis
            
                ran.observation, ran.classLabels = labelNodes(RAN=ran)
                tpath='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/socialite/data4/sleep-data-student-wise/estest'
                all_files = glob.glob(os.path.join(tpath, "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent
                allAveg = []
                rprt=[]
                rprt.append(ran.classLabels)
                rprt.append('[Subject-Active, Subject-Inactive]')
                for tpth in all_files:
                    ran.intermediateData1=[]#initializing intemediate test data
                    tdata,hdr = processInputCSVFile(tpth, normalize=True)
                    ran.testData = tdata
                    rprt.append('--------------------------------------------------------------------------------------------------------------------------')
                    rprt.append(tpth)
                    # ran.testData=sklearn.preprocessing.normalize(testD/stdDev)# trainD #initializing RAN with sorted training data whitening the test data
                    ran.intermediateData1.append(ran.testData)
                    ran.propagateUp(ran, isConvex=False)
                    tdlist=[]
                    for tx in ran.intermediateData1[ran.numberOfLayers]:
                        frame=numpy.zeros(ran.intermediateData1[ran.numberOfLayers].shape[1])
                        frame[numpy.argmax(tx)]=1
                        tdlist.append(frame)
                    tdlist=numpy.asarray(tdlist)
                    avg=numpy.average(tdlist,axis=0)
                    allAveg.append(avg)
                    propagatedData=ran.intermediateData1[ran.numberOfLayers]
                    avgConfidence=numpy.average(propagatedData,axis=0)
                    rprt.append('Percentage of being [Active , Inactive]')
                    rprt.append(avg)
                    rprt.append('confidence of represenatation')
                    rprt.append(avgConfidence)
                    rprt.append('--------------------------------------------------------------------------------------------------------------------------')
        
                numpy.savetxt("/home/rahul/Dropbox/MyWork/RAN/Oberservations/report.txt", rprt,
                          delimiter="\n", fmt="%s")
        

                '''
                # ----------------------------------------------------------------------------------------------------------

                # --------------------------------------------------------------------------------------------------------
                # validating by upward propagation
                #path_test = "/home/rahul/Dropbox/RahulSharma/PHD/DataSets/Gas_Sensor_data_for_home_activity_monitoring/reduced_train1_data.csv"
                #ran.testData = processInputCSVFile(path_test)
                testD, sortedTestData, sortedTestlabel, unique_label, labelFrequency = processData(data=X_test)  # processing training data
                ran.testData=testD
                #ran.testData=sklearn.preprocessing.normalize(testD/stdDev)# trainD #initializing RAN with sorted training data whitening the test data
                ran.intermediateData1.append(ran.testData)
                ran.propagateUp(ran,isConvex=isConvex)

                    # --------------------------------------------------------------------------------------------------------
                print ("x")
                '''
                temp=[]
                ctr=0
                start=0
                for x in ran.label_frequency:
        
                    #ct =[[0]*(ran.intermediateData1[ran.intermediateData1.__len__()-1].shape[1])+[0]]
                    ct=numpy.asarray([[0.0]*(ran.intermediateData[ran.intermediateData.__len__()-1].shape[1])+[0.0]])#
                    ct = numpy.squeeze(ct)
        
                    #ct=numpy.zeros(ran.intermediateData1[ran.intermediateData1.__len__()-1]+1)#
                    for y in range(0,x):
                        i=ran.intermediateData.__len__() - 1
                        j=start
                        k=ran.intermediateData[i][j]
                        l=ran.intermediateData[i][j].argmax()
                        ct[ran.intermediateData[ran.intermediateData.__len__()-1][start].argmax()] += 1
                        start += 1
        
                    #start=x
                    ct[ct.size-1]=ran.unique_labels[ctr]
                    ctr+=1
                    temp.append(ct)
                ran.observation=temp
                '''
                ran.observation,ran.classLabels=labelNodes(RAN=ran)
                confidence_indicator=[]#it indicated the confidence of represenatation
                true_label=Y_test*(ran.labelCount-1)
                mappedTrueLabel=[]
                for j in true_label.astype(int):
                    mappedTrueLabel.append(int(ran.classLabels[j]))
                observedTestLabels=[]
                observedTestLabels1 = []
                cfdValofTstLable=[] #confidence value of test labels
                cfdValofTstLable1 = []  # confidence value of test labels
                for k in ran.intermediateData1[ran.intermediateData1.__len__()-1]:
                    confidence_indicator.append(numpy.divide(k-k.min(), k.max()-k.min()))# minimum is min() value in the array #scale every instance of propagated data between [1 0] using minmax relation
                    #confidence_indicator.append(numpy.divide(k,k.max()))# minimum min() is fixed to 0 #scale every instance of propagated data between [1 0] using minmax relation
                    observedTestLabels.append(numpy.argmax(k))
                    #k1=(1-ran.rFactor)*k + (ran.rFactor)*ran.regulate(ran.weightCoOccurance[ran.intermediateData1.__len__()-1],k)
                    #observedTestLabels1.append(numpy.argmax(k1))
                    cfdValofTstLable.append(numpy.max(k))
                    #cfdValofTstLable1.append(numpy.max(k1))
                targetnams=[]
                ctrr=0
                for l in range(0, ran.intermediateData1[ran.intermediateData1.__len__()-1].shape[1]):
                #for l in range(0,numpy.unique(ran.classLabels).size):
                    tem='Class- %s' %ctrr
                    targetnams.append(tem)
                    ctrr+=1
                print ("accuracy: %s" %(accuracy_score(mappedTrueLabel, observedTestLabels)))
                acc[5].append(accuracy_score(mappedTrueLabel, observedTestLabels))
                try:
                    print(classification_report(mappedTrueLabel, observedTestLabels, target_names=targetnams, digits=7))
                except IndexError:
                    pass


                reports.append(classification_report(mappedTrueLabel, observedTestLabels, target_names=targetnams,digits=7))
                #uncoment to put non-convex weight in report

                if isConvex==False:
                    reports.append('-------RANs non-convex-weights------')
                    for ets in ran.nConvexWeights[1]:
                        reports.append("%s" %ets)
                    reports.append('--------------------------------')

                yp, yr, yf1, ys = sklearn.metrics.precision_recall_fscore_support(mappedTrueLabel, observedTestLabels)
                # logic for building binary weight based upon f1 score
                yfone = []
                for f1i in yf1:
                    if f1i == 0:
                        yfone.append(0.0)
                    else:
                        yfone.append(1.0)
                #ys = yfone
                y_test=numpy.asarray(observedTestLabels)

                #logic to calculate score of instance based upon each identified category
                #here we first scale every instance of propagated data between [1 0] using minmax relation as confidence_indicator
                #we then fing an average of real activation and confidence_indicators to obtain a score
                y_score=[]

                #initializing lists needed for plotting
                auc=[]#[-1]*numpy.unique(mappedTrueLabel).size#label_count
                tp=[]#[-1]*numpy.unique(mappedTrueLabel).size#label_count
                fp=[]#[-1]*numpy.unique(mappedTrueLabel).size#label_count

                confidence_indicator=numpy.asarray(confidence_indicator)
                actual_activation=ran.intermediateData1[ran.intermediateData1.__len__()-1]
                for x in range(0,ran.intermediateData1[ran.intermediateData1.__len__()-1].shape[1]):
                    y_score.append(numpy.divide(confidence_indicator.T[x]+actual_activation.T[x],float(ran.intermediateData1[ran.intermediateData1.__len__()-1].shape[1])))
                y_test_uqlabels,freq =numpy.unique(y_test,return_counts=True)

                #logic to generate ROC curve for all node in top most layer
                uniqueMTL=numpy.unique(mappedTrueLabel) #unique mapped true labels
                binarizedMTL=[]#binarized mapped true label
                tctr=0#counter

                for tx in uniqueMTL:
                    mtl=[]
                    for tx1 in mappedTrueLabel:
                        if tx1==tx:
                            mtl.append(1)
                        else:
                            mtl.append(0)
                    binarizedMTL.append(mtl)
                    try:
                        fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy.asarray(mtl), y_score[tctr],pos_label=1)
                        #auc[tx]=sklearn.metrics.auc(fpr, tpr)
                        auc.append(sklearn.metrics.auc(fpr, tpr))
                        #if uniqueMTL.size == label_count:# accumulate auc values when it size matches with label count
                            #AUC.append(auc)
                        #tp[tx]=tpr
                        #fp[tx]=fpr
                        tp.append(tpr)
                        fp.append(fpr)
                        tctr+=1
                        #plotROC(fpr=fp, tpr=tp,path="../Oberservations//pass%s/ROC-pass-%s.png" %(iter1, pltctr),auc=auc)
                    except ValueError:
                        pass
                AUC.append(auc)
                plotROC(fpr=fp, tpr=tp,path="../Oberservations/Iter%s/pass%s/ROC-pass-%s.png" % (topiter,iter1, pltctr),auc=auc)
                #old logic of ROC plotting
                '''
                if ran.classLabels[0]==0:
        
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy.asarray(mappedTrueLabel), y_score[1],
                                                            pos_label=ran.classLabels[1])
                else:
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy.asarray(mappedTrueLabel), y_score[1],
                                                                pos_label=ran.classLabels[0])
                auc.append(sklearn.metrics.auc(fpr,tpr))
                fprCurve1.append(fpr);fp.append(fpr)
                tprCurve1.append(tpr);tp.append(tpr)
                #plotROC(fpr=fpr,tpr=tpr,path="../Oberservations/roc-1-pass-%s.png" %pltctr,auc=sklearn.metrics.auc(fpr,tpr))
                if ran.classLabels[0]==0:
        
                    fpr,tpr,threshold=sklearn.metrics.roc_curve(numpy.asarray(mappedTrueLabel),y_score[0],pos_label=ran.classLabels[0])
                else:
                    fpr, tpr, threshold = sklearn.metrics.roc_curve(numpy.asarray(mappedTrueLabel), y_score[0],
                                                                pos_label=ran.classLabels[1])
                fprCurve.append(fpr);fp.append(fpr)
                tprCurve.append(tpr);tp.append(tpr)
        
                auc.append(sklearn.metrics.auc(fpr, tpr))
                plotROC(fpr=fp, tpr=tp,
                    path="../Oberservations/roc-pass-%s.png" % pltctr,
                    auc=auc)
                '''
                try:
                    prc,rrc,tres=sklearn.metrics.precision_recall_curve(numpy.asarray(mappedTrueLabel),y_score[0],pos_label=ran.classLabels[0])

                    prCurve.append(prc)
                    rCurve.append(rrc)
                    '''
                    #-------------------plotting parallel coordinates----for UCIHAR data------------------------------------------------------------------
                    fig3,ax1=plt.subplots()
            
                    plotData = numpy.asarray(ran.intermediateData1[ran.intermediateData1.__len__() - 1].transpose().tolist() + [
                    numpy.asarray(observedTestLabels)]).transpose()
                    lst=['Motion State','Stationary State']
                    tl=numpy.unique(observedTestLabels)
                    Pdata=pandas.DataFrame(plotData,columns=['Node-0','Node-1','l'])
                    Pdata['l'].replace(to_replace=tl, value=lst, inplace=True)
                    groupedPdata=Pdata.groupby('l')
                    grp1=groupedPdata.get_group(lst[0])
                    grp2=groupedPdata.get_group(lst[1])
            
                    ax1.set_ylim(0, 1)
                    ax1.set_ylabel("Activation", rotation=90, fontsize=10)
                    pandas.tools.plotting.parallel_coordinates(grp1, 'l',color=['b'])
                    fig3.savefig("/home/rahul/Dropbox/MyWork/RAN/Oberservations/PC_Motion.png")
                    plt.close(fig3)
            
                    fig4, ax2 = plt.subplots()
                    ax2.set_ylim(0, 1)
                    ax2.set_ylabel("Activation", rotation=90, fontsize=10)
                    pandas.tools.plotting.parallel_coordinates(grp2, 'l',color=['g'])
                    fig4.savefig("../Oberservations/PC_Stationary.png")
                    plt.close(fig4)
            
                    plotData1= numpy.asarray(ran.intermediateData1[ran.intermediateData1.__len__() - 1].transpose().tolist() + [
                    numpy.asarray(true_label)]).transpose()
                    lst1 = ['Walking', 'Walking-Upstairs', 'Walking-Downstairs', 'Sitting', 'Standing', 'Laying']
                    tl1 = numpy.unique(true_label)
                    Pdata1 = pandas.DataFrame(plotData1, columns=['Node-0', 'Node-1', 'l1'])
                    Pdata1['l1'].replace(to_replace=tl1, value=lst1, inplace=True)
                    groupedPdata1 = Pdata1.groupby('l1')
                    c=['b','g','r','c','m','y']
                    ctx=0
                    for name,grp in groupedPdata1:
                        f,ax=plt.subplots()
                        ax.set_ylim(0,1)
                        ax.set_ylabel("Activation", rotation=90, fontsize=10)
                        pandas.tools.plotting.parallel_coordinates(grp, 'l1',color=c[ctx])
                        f.savefig("../Oberservations/PC_%s.png" %name)
                        plt.close(f)
                        ctx+=1
                    fig5,ax5=plt.subplots()
                    ax5.set_ylim(0,1)
                    ax5.set_ylabel("Activation", rotation=90, fontsize=10)
                    grp1=pandas.concat([groupedPdata1.get_group(lst1[0]),groupedPdata1.get_group(lst1[1]),groupedPdata1.get_group(lst1[2])])
                    pandas.tools.plotting.parallel_coordinates(grp1, 'l1',color=['b','g','r'])
                    fig5.savefig("/home/rahul/Dropbox/MyWork/RAN/Oberservations/PC_Mstate.png")
                    plt.close(fig5)
            
                    fig6, ax6 = plt.subplots()
                    ax6.set_ylim(0, 1)
                    ax6.set_ylabel("Activation", rotation=90, fontsize=10)
                    grp2 = pandas.concat(
                        [groupedPdata1.get_group(lst1[3]), groupedPdata1.get_group(lst1[4]), groupedPdata1.get_group(lst1[5])])
                    pandas.tools.plotting.parallel_coordinates(grp2, 'l1',color=['c','m','y'])
                    fig6.savefig("../Oberservations/PC_Sstate.png")
                    plt.close(fig6)
                    #----------------------------------------------------------------------------------------------------------
                    '''
                    thresCurve.append(tres)


                    pr[5].append(numpy.average(yp,weights=ys))
                    rec[5].append(numpy.average(yr,weights=ys))
                    fone[5].append(numpy.average(yf1,weights=ys))

                    # logic for building binary weight based upon f1 score
                    for xf1 in fone[5]:
                        if xf1 == 0:
                            foneWeight[5].append(0.0)
                        else:
                            foneWeight[5].append(1.0)


                    sup[5].append(numpy.sum(ys))
                    modelArc.append(ran.NumberOfNodes)
                    reports.append("nodes count in each layer: %s" %ran.NumberOfNodes)
                    reports.append("Class Labels: %s" %ran.classLabels)
                    reports.append("Accuracy: %s" % accuracy_score(mappedTrueLabel, observedTestLabels))
                    # Compute confusion matrix
                    cnf_matrix = confusion_matrix(mappedTrueLabel, observedTestLabels)
                    numpy.set_printoptions(precision=2)


                    # Plot normalized confusion matrix

                    plot_confusion_matrix(cnf_matrix, classes=numpy.asarray(targetnams),
                                          title='confusion matrix')
                    #plt.show()
                    #fig2 = plt.figure(2)
                    f2=plt.gcf()
                    f1 = plt.gca()
                    f1.xaxis.set_label_coords(0.50, -0.01)
                    f1.yaxis.set_label_coords(-0.02, 0.50)
                    #f1.xaxis.set_label_text("Predicted Label",)

                    f2.set_size_inches(4,3)

                    #f1.set_xlabel("")

                    plt.savefig("../Oberservations/Iter%s/pass%s/CM-pass-%s.png" %(topiter,iter1,pltctr))

                    #plt.show()
                    plt.close()
                except ValueError:
                    pass
                pltctr += 1
                '''
                #------------testing user input- downwards--------------------
                stc = 0
        
                while stc != 1:  # loop until test inputs finishes
                    testdata = [];
                    tempx = []
                    stc2 = 1
                    while stc2 != 0:  # loop unitl correct inputs are provided
                        try:
                            cx = float(raw_input('enter regulation factor [0-1]: '))
        
        
                        except ValueError:
                            stc2 = 1
                        else:
                            if cx < 0 or cx > 1:
                                stc2 = 1
                                print "enter value in range [0 1]\n"
                            else:
                                ran.rFactor = cx
                                stc2 = 0
                    for data1 in range(0, ran.intermediateData[len(ran.weight)].shape[1]):
                        stc1 = 1
                        while stc1 != 0:  # loop unitl correct inputs are provided
                            try:
                                c = float(raw_input('enter value %s' % data1 + ':'))
        
        
                            except ValueError:
                                stc1 = 1
                            else:
                                if c < 0 or c > 1:
                                    stc1 = 1
                                    print "enter value in range [0 1]\n"
                                else:
                                    testdata.append(c)
                                    stc1 = 0
        
                    tempx.append(testdata)
                    ran.testData = numpy.asarray(tempx)
                    ran.intermediateTestData[len(ran.weight)] = ran.testData
                    ran.train_RAN_Downward(ran, ran.intermediateTestData)
        
                    #d1 = pandas.read_csv("../data/iris1.csv")
                    #parallel_coordinates(d1, 'label')
        
                    #d = numpy.squeeze(ran.intermediateDataObtained[0]) * numpy.asarray([7.9, 4.4, 6.9, 2.5])  # rescaling the valuse in the original for for visualization
                    #d = ran.intermediateDataObtained[0] * numpy.asarray([7.9, 4.4, 6.9, 2.5])  # rescaling the valuse in the original for for visualization
                    # d=numpy.asarray(d).tolist()
                    #ran.dataconversion(d, "../data/centroids.csv", ran.weight[0].shape[1], label="obervation")
                    #d1 = pandas.read_csv("../data/centroids.csv")
        
                    #parallel_coordinates(d1, 'label', color='k')
                    plt.show()
                    stop = 1
                    while stop == 1:
                        try:
                            stc = int(raw_input('Enter 1 to stop iteration'))
                            stop = 0
                        except ValueError:
                            stop = 1
        
                #-------------------------------------------
                '''
                reports.append("------------------------------------------------------------------------------------------------------------------------------")
                iter+=1
                #r.append(ran)
            try:
                avg_auc=numpy.average(AUC,axis=0)#calculating average AUC for all the passes
                #converting average auc into string
                str=""
                for iauc in avg_auc:
                    str=str+"%s\t"%iauc
                reports.append("Average AUC for each class: %s "%str)
                report_AUC.append("Av-AUC-RD-%s\t%s"%(iter1,str))
            except Exception as e:
                print (e)
                report_AUC.append(e)
                pass

            reports.append("Model\taverage precision\tSD-Precision\taverage recall\tSD-Recall\taverage F1\tSD-F1\taverage support\taverage accuracy\tSD-Accuracy")
            for ind in range(0,pr.__len__()):

                averages="RD-%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" %(iter1,modelName[ind],numpy.average(pr[ind]),numpy.std(pr[ind],ddof=1),numpy.average(rec[ind]),numpy.std(rec[ind],ddof=1),numpy.average(fone[ind]),numpy.std(fone[ind],ddof=1),numpy.average(sup[ind]),numpy.average(acc[ind]),numpy.std(acc[ind],ddof=1))
                reports.append(averages)
                report1.append(averages)

            # saving the report
                avPrecision[ind].append(numpy.average(pr[ind]))
                avRecall[ind].append(numpy.average(rec[ind]))
                avF1[ind].append(numpy.average(fone[ind]))
                avAccuracy[ind].append(numpy.average(acc[ind]))

            numpy.savetxt("../Oberservations/Iter%s/pass%s/Modelcomparison.txt"%(topiter,iter1), reports, delimiter="\n", fmt="%s")
        report1.append("------------------iteration %s ends------------" % topiter)
    numpy.savetxt(
        "../Oberservations/comparitiveReport.txt",
        report1, delimiter="\n", fmt="%s")
    numpy.savetxt(
        "../Oberservations/AUCReport.txt",
        report_AUC, delimiter="\n", fmt="%s")
    print ("x")


if __name__ == "__main__":
    main()