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
from pandas.tools.plotting import parallel_coordinates

def main():



    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/CreditApproval/RansForm.csv' # identify category with k=2
    path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/CreditApproval/german.csv' #german credit data k=2
    #path='/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/newDataToExplore/new/bank/RansForm.csv' #identify categories with k=2

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
    avPrecision=[[], [], [], [], []];avRecall=[[], [], [], [], []];avF1=[[], [], [], [], []];avAccuracy=[[], [], [], [], []]
    listPrecision = [[], [], [], [], []];listRecall = [[], [], [], [], []];listF1 = [[], [], [], [], []];listAccuracy = [[], [], [], [], []]
    #loop for number of iterations of the experiment
    for topiter in range(0, 30): #loop 30 times
        iter1 = 0

        tr_size = 0.5  # size of train data
        #loop for the 9 Research Designs of the experiment
        for iter1 in range(1,2):
            r = []
            labelReport=[]
            reports = []
            acc = [[], [], [], [], []]
            pr = [[], [], [], [], []]
            rec = [[], [], [], [], []]
            fone = [[], [], [], [], []]
            sup = [[], [], [], [], []]
            foneWeight= [[], [], [], [], []]
            lstPrecision = [[], [], [], [], []];lstRecall = [[], [], [], [], []];lstF1 = [[], [], [], [], []];lstAccuracy = [[], [], [], [], []]
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
                print "x"
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
                #---------------------------------------------------------------------------------------------
                ## logic to normalize labels to integer in general should be commented when experimenting
                #  with Glassidneitfication, and UCIHAR data

                Y_train_temp = (Y_train * 2).astype(int)
                Y_test_temp = (Y_test * 2).astype(int)
                labelReport.append(Y_test_temp)

                reports.append("----------------------------------Pass %s---------------------------------------------------------------" % iter)

                # -------MLP training
                reports.append("-------------------Multi-layer perceptron model-----------------------")
                modelName.append("MLP")
                print "-------------------------MLP----------------------------------"
                mlp = MLPClassifier( random_state=1,
                                        hidden_layer_sizes=(5), max_iter=200)
                mlp.fit(X_train, Y_train_temp)
                predictionMlp = mlp.predict(X_test)
                labelReport.append(predictionMlp)
                print(confusion_matrix(Y_test_temp, predictionMlp))
                #cfx=confusion_matrix(Y_test_temp, predictionMlp)
                reports.append("Confusion Matrix")
                for cf in confusion_matrix(Y_test_temp, predictionMlp).tolist():
                    reports.append(str(cf))

                print(classification_report(Y_test_temp, predictionMlp,digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, predictionMlp))
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
                print "-------------------------Logistic Regression----------------------------------"

                modelName.append("Logistic Regresion")
                logreg=LogisticRegression(multi_class='multinomial',solver='newton-cg',max_iter=5,class_weight='balanced',C=0.00001)
                mLogreg=logreg.fit(X_train,Y_train_temp)
                pred=mLogreg.predict(X_test)
                labelReport.append(pred)
                print(confusion_matrix(Y_test_temp, pred))
                reports.append("Confusion Matrix")
                for cf in confusion_matrix(Y_test_temp, pred).tolist():
                    reports.append(str(cf))
                print(classification_report(Y_test_temp, pred,digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, pred))

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
                print "------------------------- rbm----------------------------------"
                reports.append("-------------------- RBM-----------------------------")
                modelName.append("RBM")
                rbm=BernoulliRBM()
                #logistic = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=10,class_weight='balanced', C=1)
                logistic=LogisticRegression(multi_class='multinomial',solver='newton-cg',C=1)
                classifire=Pipeline(steps=[('rbm',rbm),('logistic',logistic)])
                rbm.learning_rate=0.001
                rbm.n_iter=200
                rbm.n_components=10
                #logistic.C=6000.0

                #with multiple classes
                #logistic.fit(X_train, (Y_train * 6).astype(int))
                #classifire.fit(X_train,(Y_train*6).astype(int))

                #with two classes
                logistic.fit(X_train, Y_train_temp)
                classifire.fit(X_train,Y_train_temp)

                prd=classifire.predict(X_test)
                labelReport.append(prd)

                # with multiple classes
                #print(confusion_matrix((Y_test * 6).astype(int), prd))
                #print(classification_report((Y_test * 6).astype(int), prd))

                # with two classes
                print(confusion_matrix(Y_test_temp, prd))
                reports.append("Confusion Matrix")
                for cf in confusion_matrix(Y_test_temp, prd).tolist():
                    reports.append(str(cf))
                print(classification_report(Y_test_temp, prd,digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prd))
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
                print "-------------------------K-Neareast Neoghbor----------------------------------"
                #reports.append("--------------------------K-Neareast Neoghbor-------------------------")
                modelName.append("K-NN")
                clf=neighbors.KNeighborsClassifier(n_neighbors=30)#,weights='distance',leaf_size=5,algorithm='auto',)
                clf.fit(X_train,Y_train_temp)
                prdt=clf.predict(X_test)
                labelReport.append(prdt)
                print(confusion_matrix(Y_test_temp, prdt))
                reports.append("Confusion Matrix")
                for cf in confusion_matrix(Y_test_temp, prdt).tolist():
                    reports.append(str(cf))
                print(classification_report(Y_test_temp, prdt,digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prdt))
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
                print "-------------------------SGD Classifier----------------------------------"
                reports.append("-------------------------SGD Classifier---------------------------")
                modelName.append("SGD")
                sgd = SGDClassifier(alpha=0.0001,n_iter=5,epsilon=0.25)
                sgd.fit(X_train, Y_train_temp)
                prdtt = sgd.predict(X_test)
                print(confusion_matrix(Y_test_temp, prdtt))
                reports.append("Confusion Matrix")
                for cf in confusion_matrix(Y_test_temp, prdtt).tolist():
                    reports.append(str(cf))
                print(classification_report(Y_test_temp, prdtt,digits=7))

                print "accuracy: %s" % (accuracy_score(Y_test_temp, prdtt))
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
    print "x"


if __name__ == "__main__":
    main()