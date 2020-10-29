__author__ = 'Rahul'
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF, OWL
from rdflib import Graph
from rdflib import Namespace
import numpy


#method to add layer Individual
def addLayer(g,layer, n, nodeList, layerId):
    g.add((layer, RDF.type, n.Layer))
    g.add((layer, RDF.type, OWL.NamedIndividual))
    g.add((layer, n.layerId, Literal(layerId)))
    for node in nodeList:
        g.add((layer, n.hasNodes, node))

#method to add weight indiviuual
def addWeight(g,w,n,wt, targetNode):
    g.add((w, RDF.type, n.Layer))
    g.add((w, RDF.type, OWL.NamedIndividual))
    g.add((w, n.weight, Literal(wt)))
    g.add((w, n.hasTargetNode, targetNode))

#method to add node individual
def addNode(g,node, n, weightList=None, nodeId=None, layerUri=None):
    g.add((node, RDF.type, n.Layer))
    g.add((node, RDF.type, OWL.NamedIndividual))
    g.add((node, n.inLayer, layerUri))
    g.add((node, n.nodeId, Literal(nodeId)))
    if weightList!=None:
        g.add((node, n.hasWeight, weightList))


def build_rdf(lst=None,file_name=None):

    lyrUri=[] #list storing the URI for layer
    lyrNodes=[] #list storing the URI for nodes
    nodesUri=[] #list to store URI of nodes
    weightUri=[]
    newlas=[]# list to store nodes in each layer
    g = Graph()
    weightCount=0
    n = Namespace("http://www.semanticweb.org/rahul"
              "/ontologies/2015/5/transferlearning#")
    #how many layers
    layers= lst.__len__()
    nodeCount=0




    #number of nodes in the layer
    for lyr in range(0, layers):
        #number of nodes in the layer below
        LNodes=lst[lyr].shape[0]
        lt=[]
        s = "http://www.semanticweb.org/rahul" \
        "/ontologies/2015/5/transferlearning#layer%s" %lyr
        s= URIRef(s)
        lyrUri.append(s) #appending the layer URIs to the list
        for x in range(0,LNodes):
            s1= "http://www.semanticweb.org/rahul" \
            "/ontologies/2015/5/transferlearning#node%s" %nodeCount
            s1=URIRef(s1)
            nodeCount+=1
            #print s1
            lt.append(s1)
            nodesUri.append(s1) #appending the Node URIs to the list
        newlas.append(lt) #appending the layer wise list of nodes

    #for last layer

    lnode=lst[layers-1].shape[1]
    lt=[]
    s= "http://www.semanticweb.org/rahul" \
   "/ontologies/2015/5/transferlearning#layer%s" %(layers)
    s= URIRef(s)
    lyrUri.append(s)
    for x in range(0,lnode):
            s1= "http://www.semanticweb.org/rahul/" \
            "ontologies/2015/5/transferlearning#node%s" %nodeCount
            s1=URIRef(s1)
            nodeCount+=1
            #print s1
            lt.append(s1)
            nodesUri.append(s1)
    newlas.append(lt)
    print newlas


    count=0
    #creating layer individuals
    for i in range(0, len(lyrUri)):
        addLayer(g,lyrUri[i],n,newlas[i],i)
        #creating node individuals


    #creating individuals for weight and nodes

    for lyr in range(0,layers):
        x=lst[lyr]
        thisL=newlas[lyr]
        aboveL=newlas[lyr+1]
        #itemp=newlas[lyr].__len__()
        for i in range(0, newlas[lyr].__len__()):
            #jtemp=newlas[lyr+1].__len__()
            sourceNode=thisL[i]
            #temp= str(thisL[i])
            for j in range(0, newlas[lyr+1].__len__()):
                destNode= aboveL[j]
                #temp1= aboveL[j]
                weight = x[i][j]
                ws="http://www.semanticweb.org/rahul/ontologies" \
               "/2015/5/transferlearning#weight%s" %weightCount
                weightCount+=1
                ws=URIRef(ws)
                weightUri.append(ws)
                addNode(g,sourceNode,n,ws,i,lyrUri[lyr])
                addWeight(g,ws,n,weight,destNode)

    #creating individual for the last layer nodes
    lnode=lst[layers-1].shape[1]
    #creating nodes for last layer
    for i in range(0,len(newlas[lnode])):
        addNode(g,newlas[lnode][i],n,None,i,lyrUri[lnode])
    g.bind("owl", OWL)
    g.bind(" ", n)
    name= file_name+".owl"
    print g.serialize(name,format='turtle')


#list storing the weight array in 2d
lst= [numpy.asarray([[1.0,2.0],[1.0,2.0],[2.0,3.0],[3.0,4.0]]),numpy.asarray( [[1.0,2.0],[2.0,3.0]]),numpy.asarray( [[1.0,2.0],[2.0,3.0]])]
file_name="name1"
build_rdf(lst=lst,file_name=file_name)

print "*"

