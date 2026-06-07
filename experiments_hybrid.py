import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def erdos_renyi_hybrid_kendalltau(n, p, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_hybrid_kendalltau(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_hybrid_kendalltau(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def erdos_renyi_hybrid_spearman(n, p, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_hybrid_spearman(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_hybrid_spearman(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.hybrid, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

##hybrid
##RESULTS
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-KENDALL-TAU
##Mean tau: 0.13662357805262856
##SD tau: 0.03378977505231926
##Combined p-value: 0.0
##Cohen's d tau: 4.043340858028323
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-KENDALL-TAU
##Mean tau: 0.29849007681899203
##SD tau: 0.036797634376005306
##Combined p-value: 0.0
##Cohen's d tau: 8.11166483608601
##--------------------------------------------------
##GEOMETRIC-GRAPHS-KENDALL-TAU
##Mean tau: 0.07328044179285138
##SD tau: 0.05319577603125921
##Combined p-value: 0.0
##Cohen's d tau: 1.3775612888848525
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-SPEARMAN
##Mean rho: 0.16273752358976512
##SD rho: 0.05576834364088787
##Combined p-value: 0.0
##Cohen's d rho: 2.918098565696871
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-SPEARMAN
##Mean rho: 0.39693634575796943
##SD rho: 0.04980703647290178
##Combined p-value: 0.0
##Cohen's d rho: 7.9694833073220135
##--------------------------------------------------
##GEOMETRIC-GRAPHS-SPEARMAN
##Mean rho: 0.10773931266289298
##SD rbo: 0.06111538503606743
##Combined p-value: 0.0
##Cohen's d rho: 1.7628836437717001

mu,sd,d,p=erdos_renyi_hybrid_kendalltau(
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

mu,sd,d,p=core_periphery_hybrid_kendalltau(
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

mu,sd,d,p=geometric_hybrid_kendalltau(
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

mu,sd,d,p=erdos_renyi_hybrid_spearman(
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

mu,sd,d,p=core_periphery_hybrid_spearman(
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

mu,sd,d,p=geometric_hybrid_spearman(
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
