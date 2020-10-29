__author__ = 'Rahul'

from rdflib import URIRef, BNode, Literal
from rdflib.namespace import RDF, FOAF, OWL
from rdflib import Graph
from rdflib import Namespace
import numpy

g= Graph()

#retrieving and parsing the ontology tl.owl storen on web
#g.parse("http://tlexp.site88.net/name.owl", format="turtle")
g.parse("/home/rahul/Dropbox/MyWork/RAN_CRBMv1.0/RAN/name.owl", format="turtle")
#query the ontology
q = g.query(
        """PREFIX     : <http://www.semanticweb.org/rahul/ontologies/2015/5/transferlearning#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?w
            WHERE{ ?layer :hasNodes ?node.
                   ?layer :layerId "1"^^xsd:positiveInteger.
                   ?node :nodeId "1"^^xsd:positiveInteger.
                   ?node :hasWeight ?weight.
                   ?weight :hasTargetNode ?nodes.
                   ?nodes :nodeId "1"^^xsd:positiveInteger.
                   ?weight :weight ?w.}"""
)

#printing the result of the query
print q
for row in q:
    print row[0]

print "x"