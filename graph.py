import math
import numpy as np
import networkx as nx

class Graph:
    def __init__(self, n):
        #Initiates an undirected graph with vertex labels 0,1,...,n-1
        self.vertex_count=n
        self.adjmatrix=[]
        for i in range(n):
            adj=[0 for j in range(n)]
            self.adjmatrix.append(adj)

    def set_adj_matrix(self, matrix):
        #Sets the adjacency matrix of the graph.
        self.adjmatrix=matrix

    def isedge(self, v1, v2):
        #Returns if there is an edge between vertices labeled v1 and v2
        return self.adjmatrix[v1][v2]==1

    def addedge(self, v1, v2):
        #Adds an undirected edge between the vertices labeled v1 and v2
        self.adjmatrix[v1][v2]=1
        self.adjmatrix[v2][v1]=1

    def removeedge(self, v1, v2):
        #Removes an undirected edge between the vertices labeled v1 and v2
        self.adjmatrix[v1][v2]=1
        self.adjmatrix[v2][v1]=1

    def get_adjacent(self, v1):
        #Gets all adjacent vertices to the vertex labeled v1
        adj=[]
        for j in range(n):
            if self.adjmatrix[v1][j]==1:
                adj.append(j)
        return adj

    def spectral(self):
        #Returns the spectral radius and perron eigenvector of the graph.
        adj=np.array(self.adjmatrix)
        vals, vecs=np.linalg.eigh(adj)
        max_index=np.argmax(vals)
        perron=vecs[:,max_index]
        if perron[0]<0:
            perron=-1*perron
        return vals[max_index], perron

    def get_cores(self):
        adj=np.array(self.adjmatrix)
        g=nx.from_numpy_array(adj)
        core_nums=nx.core_number(g)
        return [core_nums[i] for i in range(self.vertex_count)]

    def get_degrees(self):
        adj=np.array(self.adjmatrix)
        g=nx.from_numpy_array(adj)
        return g.degree()
