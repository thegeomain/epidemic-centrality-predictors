import random
import numpy as np
from graph import Graph

class SIR:
    def __init__(self, sim_graph, initial_vertices, initial_recovered, beta, gamma):
        #beta is probability of infected person infecting neighbor in a given time step, gamma is probability of recovery in a given time step
        #initial_vertices is the list of vertices that start infected, initial_recovered are the "vaccinated" vertices
        self.graph=Graph(sim_graph.vertex_count)
        self.graph.set_adj_matrix(sim_graph.adjmatrix)
        self.vertex_count=self.graph.vertex_count
        self.beta=beta
        self.gamma=gamma
        self.statuses=["S" for i in range(self.graph.vertex_count)]
        for i in initial_vertices:
            self.statuses[i]="I"
        for i in initial_recovered:
            self.statuses[i]="I"

    def spread(self):
        #Spreads the infection from infected vertices to sucesptible with probabilty beta
        new_i=[]
        for i in range(self.vertex_count):
            if (self.statuses[i]=="I"):
                for j in range(self.vertex_count):
                    if self.statuses[j]=="S" and self.graph.isedge(i,j):
                        num=random.random()
                        if num<self.beta:
                            new_i.append(j)
        for i in new_i:
            self.statuses[i]="I"

    def recover(self):
        #Sets infected vertices to recover with probability gamma
        for i in range(self.vertex_count):
            if (self.statuses[i]=="I"):
                num=random.random()
                if num<self.gamma:
                    self.statuses[i]="R"

    def detect_I(self):
        #Detects whether there are any infected vertices
        for i in self.statuses:
            if (i=="I"):
                return True
        return False

    def get_counts(self):
        #Gets counts of S, I, and R respectively
        self.count_S=0
        self.count_I=0
        self.count_R=0
        for i in self.statuses:
            if i=="S":
                self.count_S+=1
            if i=="I":
                self.count_I+=1
            if i=="R":
                self.count_R+=1
        return (self.count_S, self.count_I, self.count_R)

    def time_step(self):
        #Does a time step of the SIR algorithm
        self.spread()
        self.recover()
        
