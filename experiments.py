import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def erdos_renyi_perron_kendalltau(n, p, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_perron_kendalltau(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_perron_kendalltau(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def erdos_renyi_perron_spearman(n, p, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_perron_spearman(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_perron_spearman(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.perron_square, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

##Perron Eigenvector Correlations
##RESULTS
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-KENDALL-TAU
##Mean tau: 0.13358215485651356
##SD tau: 0.03482390945207198
##Combined p-value: 0.0
##Cohen's d tau: 3.8359321787337577
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-KENDALL-TAU
##Mean tau: 0.27540086647941103
##SD tau: 0.04182326049904626
##Combined p-value: 0.0
##Cohen's d tau: 6.584873182847409
##--------------------------------------------------
##GEOMETRIC-GRAPHS-KENDALL-TAU
##Mean tau: 0.06494850467365434
##SD tau: 0.048898913332344925
##Combined p-value: 3.3861802251067274e-14
##Cohen's d tau: 1.328219795647139
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-SPEARMAN
##Mean rho: 0.16143106170418384
##SD rho: 0.05390228607363395
##Combined p-value: 0.0
##Cohen's d rho: 2.994883398519661
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-SPEARMAN
##Mean rho: 0.36329774749962807
##SD rho: 0.04783061904456721
##Combined p-value: 0.0
##Cohen's d rho: 7.595505865418919
##--------------------------------------------------
##GEOMETRIC-GRAPHS-SPEARMAN
##Mean rho: 0.10301869265572465
##SD rbo: 0.0595880326407804
##Combined p-value: 0.0
##Cohen's d rho: 1.7288486981397924

mu,sd,d,p=erdos_renyi_perron_kendalltau(
    n=300,
    p=0.03,
    beta=0.1/15,
    gamma=0.1,
    iterations=50)
print("RESULTS")
print("--------------------------------------------------")
print("ERDOS-RENYI-GRAPHS-KENDALL-TAU")
print("Mean tau:", mu)
print("SD tau:", sd)
print("Combined p-value:", p)
print("Cohen's d tau:", d)

mu,sd,d,p=core_periphery_perron_kendalltau(
    n=300,
    core_size=60,
    p_c=0.20,
    p_p=0.005,
    p_cp=0.05,
    beta=0.1/15,
    gamma=0.1,
    iterations=50)
print("--------------------------------------------------")
print("CORE-PERIPHERY-GRAPHS-KENDALL-TAU")
print("Mean tau:", mu)
print("SD tau:", sd)
print("Combined p-value:", p)
print("Cohen's d tau:", d)

mu,sd,d,p=geometric_perron_kendalltau(
    n=300,
    distance=0.10,
    beta=0.1/15,
    gamma=0.1,
    iterations=30)
print("--------------------------------------------------")
print("GEOMETRIC-GRAPHS-KENDALL-TAU")
print("Mean tau:", mu)
print("SD tau:", sd)
print("Combined p-value:", p)
print("Cohen's d tau:", d)

mu,sd,d,p=erdos_renyi_perron_spearman(
    n=300,
    p=0.03,
    beta=0.1/15,
    gamma=0.1,
    iterations=50)
print("--------------------------------------------------")
print("ERDOS-RENYI-GRAPHS-SPEARMAN")
print("Mean rho:", mu)
print("SD rho:", sd)
print("Combined p-value:", p)
print("Cohen's d rho:", d)

mu,sd,d,p=core_periphery_perron_spearman(
    n=300,
    core_size=60,
    p_c=0.20,
    p_p=0.005,
    p_cp=0.05,
    beta=0.1/15,
    gamma=0.1,
    iterations=50)
print("--------------------------------------------------")
print("CORE-PERIPHERY-GRAPHS-SPEARMAN")
print("Mean rho:", mu)
print("SD rho:", sd)
print("Combined p-value:", p)
print("Cohen's d rho:", d)

mu,sd,d,p=geometric_perron_spearman(
    n=300,
    distance=0.10,
    beta=0.1/15,
    gamma=0.1,
    iterations=30)
print("--------------------------------------------------")
print("GEOMETRIC-GRAPHS-SPEARMAN")
print("Mean rho:", mu)
print("SD rbo:", sd)
print("Combined p-value:", p)
print("Cohen's d rho:", d)
    






