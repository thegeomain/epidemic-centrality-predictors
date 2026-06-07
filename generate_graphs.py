import numpy as np
import random
from graph import Graph

def erdos_renyi(n, p):
    #Generates an erdos renyi random graph with n vertices and adding each edge with probability p
    g=Graph(n)
    for i in range(n):
        for j in range(i):
            num=random.random()
            if num<p:
                g.addedge(i,j)
    return g

def core_periphery(n, core_size, p_c, p_p, p_cp):
    #Generates a dense core and sparse periphery
    g=Graph(n)
    for i in range(core_size):
        for j in range(i):
            num=random.random()
            if num<p_c:
                g.addedge(i,j)
    for i in range(core_size, n):
        for j in range(core_size, i):
            num=random.random()
            if num<p_p:
                g.addedge(i,j)
    for i in range(core_size):
        for j in range(core_size,n):
            num=random.random()
            if num<=p_cp:
                g.addedge(i,j)
    return g

def geometric(n,d):
    points=[]
    for i in range(n):
        points.append((random.random(),random.random()))
    g=Graph(n)
    for i in range(n):
        for j in range(i):
            if (pow(points[i][0]-points[j][0],2)+pow(points[i][1]-points[j][1],2))<=pow(d,2):
                g.addedge(i,j)
    return g

def ca_netscience():
    g=Graph(379)
    f=open("ca-netscience/ca-netscience.mtx","r")
    lines=f.read().splitlines()
    for line in lines:
        a,b=line.split(" ")
        if int(a)==int(b):
            continue
        g.addedge(int(a)-1,int(b)-1)
    return g

def soc_wiki_vote():
    g=Graph(889)
    f=open("soc-wiki-Vote/soc-wiki-Vote.mtx","r")
    lines=f.read().splitlines()
    for line in lines:
        a,b=line.split(" ")
        if int(a)==int(b):
            continue
        g.addedge(int(a)-1,int(b)-1)
    return g    

def power_eris1176():
    g=Graph(1176)
    f=open("power-eris1176/power-eris1176.mtx","r")
    lines=f.read().splitlines()
    for line in lines:
        a,b=line.split(" ")
        if int(a)==int(b):
            continue
        g.addedge(int(a)-1,int(b)-1)
    return g
    
