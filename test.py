import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from scipy.stats import spearmanr, kendalltau

class Test:
    #Class to conduct tests on a particular graph.
    def __init__(self, sim_graph, beta, gamma):
        #Graph is the graph to be tested, beta and gamma are the parameters in SIR
        self.graph=Graph(sim_graph.vertex_count)
        self.graph.set_adj_matrix(sim_graph.adjmatrix)
        self.vertex_count=self.graph.vertex_count
        self.beta=beta
        self.gamma=gamma

        self.spec_rad, self.perron=self.graph.spectral()
        self.perron_square=[pow(i,2) for i in self.perron]
        self.kcores=self.graph.get_cores()
        self.degrees=[i[1] for i in self.graph.get_degrees()]
        mp=max(self.perron)
        mk=max(self.kcores)
        md=max(self.degrees)
        self.hybrid=[(self.perron[i]*self.kcores[i]*self.degrees[i])/(mp*mk*md) for i in range(self.vertex_count)]

    def sir_vertex(self, v):
        #Does a SIR simulation starting at the vertex v
        #Returns the number of iterations the epidemic lasted and the total amount of vertices infected at some point
        sir=SIR(self.graph, [v], [], self.beta, self.gamma)
        number_iterations=0
        while sir.detect_I():
            sir.time_step()
            number_iterations+=1
        return number_iterations, sir.get_counts()[2]

    def sir_simulation(self):
        #Runs a SIR simulation for all vertices
        iteration_list=[]
        infected_list=[]
        for v in range(self.vertex_count):
            it, infect=self.sir_vertex(v)
            iteration_list.append(it)
            infected_list.append(infect)
        return iteration_list, infected_list

def permutation_test_spearman(x, y, n):
    #Conducts a permutation test to determine significance
    #Returns the rho statistic and the p-value
    rho, p=spearmanr(x,y)
    sims=[]
    for i in range(n):
        a, b=spearmanr(x,np.random.permutation(y))
        sims.append(a)
    sims=np.array(sims)
    rank=np.sum(sims>=rho)+1
    return rho, rank/(n+1)

def permutation_test_kendalltau(x, y, n):
    #Conducts a permutation test to determine significance
    #Returns the tau statistic and the p-value
    tau, p=kendalltau(x,y)
    sims=[]
    for i in range(n):
        a, b=kendalltau(x,np.random.permutation(y))
        sims.append(a)
    sims=np.array(sims)
    rank=np.sum(sims>=tau)+1
    return tau, rank/(n+1)
