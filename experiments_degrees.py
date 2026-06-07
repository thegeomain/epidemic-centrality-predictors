import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def erdos_renyi_degrees_kendalltau(n, p, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_degrees_kendalltau(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_degrees_kendalltau(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_kendalltau(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def erdos_renyi_degrees_spearman(n, p, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=erdos_renyi(n, p)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def core_periphery_degrees_spearman(n, core_size, p_c, p_p, p_cp, beta, gamma, iterations):
    #Returns cohen's d for rho and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=core_periphery(n, core_size, p_c, p_p, p_cp)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

def geometric_degrees_spearman(n, distance, beta, gamma, iterations):
    #Returns cohen's d for tau and the combined p-value (based on fischer's method) for erdos renyi
    tau_list=[]
    p_list=[]
    for i in range(iterations):
        g=geometric(n, distance)
        t=Test(g, beta, gamma)
        a,b=t.sir_simulation()
        tau, p_value=permutation_test_spearman(t.degrees, b, 999)
        tau_list.append(tau)
        p_list.append(p_value)
    tau_list=np.array(tau_list)
    p_list=np.array(p_list)
    d=np.mean(tau_list)/np.std(tau_list)
    X=-2*np.sum(np.log(p_list))
    combined_p=1-chi2.cdf(X, 2*len(p_list))
    return np.mean(tau_list), np.std(tau_list), d, combined_p

##degrees
##RESULTS
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-KENDALL-TAU
##Mean tau: 0.14444055987292242
##SD tau: 0.04747092843051929
##Combined p-value: 0.0
##Cohen's d tau: 3.042716134872578
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-KENDALL-TAU
##Mean tau: 0.314641735319005
##SD tau: 0.048131139910907055
##Combined p-value: 0.0
##Cohen's d tau: 6.537176054866378
##--------------------------------------------------
##GEOMETRIC-GRAPHS-KENDALL-TAU
##Mean tau: 0.16860596669745798
##SD tau: 0.04241833264014679
##Combined p-value: 0.0
##Cohen's d tau: 3.974837203711327
##--------------------------------------------------
##ERDOS-RENYI-GRAPHS-SPEARMAN
##Mean rho: 0.18208269552009537
##SD rho: 0.061778551687012996
##Combined p-value: 0.0
##Cohen's d rho: 2.9473448397200377
##--------------------------------------------------
##CORE-PERIPHERY-GRAPHS-SPEARMAN
##Mean rho: 0.3766289607979403
##SD rho: 0.053315873332746073
##Combined p-value: 0.0
##Cohen's d rho: 7.064105626618678
##--------------------------------------------------
##GEOMETRIC-GRAPHS-SPEARMAN
##Mean rho: 0.22534749132344534
##SD rbo: 0.04170158467917114
##Combined p-value: 0.0
##Cohen's d rho: 5.4038112234137845

mu,sd,d,p=erdos_renyi_degrees_kendalltau(
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

mu,sd,d,p=core_periphery_degrees_kendalltau(
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

mu,sd,d,p=geometric_degrees_kendalltau(
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

mu,sd,d,p=erdos_renyi_degrees_spearman(
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

mu,sd,d,p=core_periphery_degrees_spearman(
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

mu,sd,d,p=geometric_degrees_spearman(
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
