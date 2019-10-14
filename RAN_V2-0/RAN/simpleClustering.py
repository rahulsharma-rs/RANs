import numpy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas
import matplotlib.colors as colors
from sklearn import cluster
import sklearn.datasets



def sCluster(data=None):
    ctr=0
    clusters=[]
    simConst=0.92
    similarity=numpy.asarray([-1]*data.shape[0],dtype=float)
    pointsTraversed=[]
    for x in data:



        if ctr in pointsTraversed:
            ctr+=1
            continue
        #plt.plot(x[0], x[1], 'ko')
        #plt.xlim(0.0, 1.0)
        #plt.ylim(0.0, 1.0)
        ctr1=0
        pTrave=[]
        for y in data:
            if ctr1==ctr:
                similarity[ctr1]=1
                ctr1+=1
            else:


                #similarity[ctr1]=numpy.subtract(1,distance.sqeuclidean(x,y)/data.shape[1])
                #similarity[ctr1]=numpy.square(numpy.subtract(1,distance.sqeuclidean(x,y)/data.shape[1]))
                #similarity[ctr1]=distance.euclidean(x,y)
                #similarity[ctr1] = numpy.square(numpy.subtract(1, ((distance.euclidean(x, y) / numpy.sqrt(data.shape[1]))**(float(1)/2))))
                #similarity[ctr1]=numpy.square(1-(distance.euclidean(x, y) / numpy.sqrt(data.shape[1])**(float(1)/2)))
                similarity[ctr1] = numpy.square(1 - (distance.sqeuclidean(x,y)/data.shape[1]) ** (float(1) / 1))



                if similarity[ctr1]>=simConst:

                    pTrave.append(ctr1)

                    #plt.plot(y[0],y[1],'r',marker='>')
                ctr1 += 1
        clusters.append([ctr]+pTrave)
        #plt.show()
        pointsTraversed=numpy.unique(numpy.asarray([ctr]+pTrave+pointsTraversed)).tolist()
        ctr+=1
        centers=[]
        for x in clusters:
            #d=numpy.mean(data[x],axis=0)
            centers.append(numpy.mean(data[x],axis=0))
        centers=numpy.asarray(centers)
    return clusters, centers


if __name__=='__main__':
    temp = pandas.read_csv("/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/data/t13data.csv", skiprows=1)
    data = temp.get_values()
    iris = sklearn.datasets.load_iris()
    X = iris.data[:, :4]
    dx = numpy.transpose(data)
    #plt.plot(dx[0], dx[1], '#F0F8FF',marker='o')
    #plt.show()
    centers,ctrs1 = sCluster(data=data)
    color=[]
    for xx in colors.cnames:
        color.append(xx)
    ct=0
    ctrs=[]
    for x1 in centers:
        temp = numpy.zeros(data.shape[1])
        for y1 in x1:
            temp=numpy.add(temp,data[y1])
            plt.plot(data[y1][0],data[y1][1],color[ct],marker='o')

        ct+=1
        temp=temp/x1.__len__()
        plt.plot(temp[0], temp[1], 'k', marker="*")

        ctrs.append(temp)
    #plt.show()

    plt.plot(dx[0], dx[1], '#F0F8FF', marker='o')
    for x2 in ctrs:
        plt.plot(x2[0],x2[1],'k',marker="*")

        # plt.show()
    plt.show()








    centroids = numpy.asarray(centers).transpose()


    plt.plot(centroids[0], centroids[1], 'ko', label='Centroids', marker='o')


    print 'c'
