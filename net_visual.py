import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau
import networkx as nx


def visual_er():
    g=erdos_renyi(50,0.1)
    plt.figure()
    g=nx.from_numpy_array(np.array(g.adjmatrix))
    nx.draw_networkx(g, with_labels=False)
    plt.show()

def vis_cp():
    g = core_periphery(50, 12, 0.6, 0.03, 0.2)
    G = nx.from_numpy_array(np.array(g.adjmatrix))

    n = 50
    n_core = 12
    n_periphery = n - n_core

    # Radii for the two circles
    r_core = 0.3
    r_periphery = 1.0

    pos = {}

    # Core nodes on inner circle
    for i in range(n_core):
        angle = 2 * np.pi * i / n_core
        pos[i] = (r_core * np.cos(angle), r_core * np.sin(angle))

    # Periphery nodes on outer circle
    for i in range(n_core, n):
        j = i - n_core
        angle = 2 * np.pi * j / n_periphery
        pos[i] = (r_periphery * np.cos(angle), r_periphery * np.sin(angle))

    plt.figure()
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
    )
    plt.axis("off")
    plt.show()

def vis_geometric():
    n=50
    d=0.2
    points=[]
    for i in range(n):
        points.append((random.random(),random.random()))
    pos=dict()
    for i in range(n):
        pos[i]=points[i]
    g=Graph(n)
    for i in range(n):
        for j in range(i):
            if (pow(points[i][0]-points[j][0],2)+pow(points[i][1]-points[j][1],2))<=pow(d,2):
                g.addedge(i,j)
    plt.figure()
    g=nx.from_numpy_array(np.array(g.adjmatrix))
    nx.draw_networkx(g, pos=pos, with_labels=False)
    plt.show()

vis_cp()
