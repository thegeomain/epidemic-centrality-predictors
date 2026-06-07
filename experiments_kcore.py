import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def erdos_renyi_kcores_kendalltau(n, p, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_kcores_kendalltau(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_kcores_kendalltau(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def erdos_renyi_kcores_spearman(n, p, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_kcores_spearman(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_kcores_spearman(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.kcores, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

##k-core index correlations
##RESULTS
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-KENDALL-TAU
##Mean tau: 0.10774899768286074
##SD tau: 0.0502626840534348
##Combined p-value: 0.0
##Cohen's d tau: 2.14371754537246
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-KENDALL-TAU
##Mean tau: 0.30732569214322664
##SD tau: 0.04454635082334738
##Combined p-value: 0.0
##Cohen's d tau: 6.89900937928574
##--------------------------------------------------
##GEOMETRIC-GRAPHS-KENDALL-TAU
##Mean tau: 0.15794144083410128
##SD tau: 0.06257036660286094
##Combined p-value: 0.0
##Cohen's d tau: 2.52422111950483
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-SPEARMAN
##Mean rho: 0.1280991107191252
##SD rho: 0.051115633776045465
##Combined p-value: 0.0
##Cohen's d rho: 2.50606519485545
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-SPEARMAN
##Mean rho: 0.37343969047165876
##SD rho: 0.05899902904834602
##Combined p-value: 0.0
##Cohen's d rho: 6.3295904440333794
##--------------------------------------------------
##GEOMETRIC-GRAPHS-SPEARMAN
##Mean rho: 0.18195713420619397
##SD rbo: 0.07002590616988094
##Combined p-value: 0.0
##Cohen's d rho: 2.5984259848743823

mu,sd,d,p=erdos_renyi_kcores_kendalltau(
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

mu,sd,d,p=core_periphery_kcores_kendalltau(
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

mu,sd,d,p=geometric_kcores_kendalltau(
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

mu,sd,d,p=erdos_renyi_kcores_spearman(
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

mu,sd,d,p=core_periphery_kcores_spearman(
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

mu,sd,d,p=geometric_kcores_spearman(
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
