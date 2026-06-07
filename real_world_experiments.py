import numpy as np
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def test_graph_kendalltau(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_kendalltau(t.perron_square, b, 999)
    return tau, p_value

def test_graph_spearman(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_spearman(t.perron_square, b, 999)
    return tau, p_value

def test_graph_kcores_kendalltau(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_kendalltau(t.kcores, b, 999)
    return tau, p_value

def test_graph_kcores_spearman(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_spearman(t.kcores, b, 999)
    return tau, p_value

def test_graph_degrees_kendalltau(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_kendalltau(t.degrees, b, 999)
    return tau, p_value

def test_graph_degrees_spearman(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_spearman(t.degrees, b, 999)
    return tau, p_value

def test_graph_hybrid_kendalltau(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_kendalltau(t.hybrid, b, 999)
    return tau, p_value

def test_graph_hybrid_spearman(g, beta, gamma):
    t=Test(g,beta,gamma)
    a,b=t.sir_simulation()
    tau, p_value=permutation_test_spearman(t.hybrid, b, 999)
    return tau, p_value

##RESULTS (Perron Eigenvector Correlations)
##--------------------------------------------------
##CA NETSCIENCE KENDALL TAU
##Tau: 0.0618074632347477
##p-value: 0.068
##--------------------------------------------------
##WIKIPEDIA VOTING KENDALL TAU
##Tau: 0.265992274519672
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK KENDALL TAU
##Tau: 0.34842187593885043
##p-value: 0.001
##--------------------------------------------------
##CA NETSCIENCE SPEARMAN
##Tau: 0.06277391737836983
##p-value: 0.115
##--------------------------------------------------
##WIKIPEDIA VOTING SPEARMAN
##Tau: 0.2838836534346215
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK SPEARMAN
##Tau: 0.4416824517402113
##p-value: 0.001
##--------------------------------------------------
##RESULTS (k-core Index Correlations)
##--------------------------------------------------
##CA NETSCIENCE KENDALL TAU
##Tau: 0.1909236136402187
##p-value: 0.001
##--------------------------------------------------
##WIKIPEDIA VOTING KENDALL TAU
##Tau: 0.3609727239095073
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK KENDALL TAU
##Tau: 0.5241153244733623
##p-value: 0.001
##--------------------------------------------------
##CA NETSCIENCE SPEARMAN
##Tau: 0.18087346290856882
##p-value: 0.001
##--------------------------------------------------
##WIKIPEDIA VOTING SPEARMAN
##Tau: 0.3789004143184187
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK SPEARMAN
##Tau: 0.5842853300213281
##p-value: 0.001
##--------------------------------------------------
##RESULTS (Degree Correlations)
##--------------------------------------------------
##CA NETSCIENCE KENDALL TAU
##Tau: 0.15048989655176534
##p-value: 0.001
##--------------------------------------------------
##WIKIPEDIA VOTING KENDALL TAU
##Tau: 0.3327071025483964
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK KENDALL TAU
##Tau: 0.5170038474619211
##p-value: 0.001
##--------------------------------------------------
##CA NETSCIENCE SPEARMAN
##Tau: 0.3089015440991835
##p-value: 0.001
##--------------------------------------------------
##WIKIPEDIA VOTING SPEARMAN
##Tau: 0.3785710793172163
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK SPEARMAN
##Tau: 0.6104103494245247
##p-value: 0.001
##--------------------------------------------------
##RESULTS (Hybrid Correlations)
##--------------------------------------------------
##CA NETSCIENCE KENDALL TAU
##Tau: 0.10296034122649003
##p-value: 0.008
##--------------------------------------------------
##WIKIPEDIA VOTING KENDALL TAU
##Tau: 0.2799720871023732
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK KENDALL TAU
##Tau: 0.36332042448727075
##p-value: 0.001
##--------------------------------------------------
##CA NETSCIENCE SPEARMAN
##Tau: 0.1319948786557211
##p-value: 0.007
##--------------------------------------------------
##WIKIPEDIA VOTING SPEARMAN
##Tau: 0.39600349546454955
##p-value: 0.001
##--------------------------------------------------
##POWER NETWORK SPEARMAN
##Tau: 0.47945816613445047
##p-value: 0.001

tau, p=test_graph_kendalltau(ca_netscience(), 0.1/15, 0.1)
print("RESULTS (Perron Eigenvector Correlations)")
print("--------------------------------------------------")
print("CA NETSCIENCE KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kendalltau(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kendalltau(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_spearman(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("CA NETSCIENCE SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_spearman(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_spearman(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_kendalltau(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("RESULTS (k-core Index Correlations)")
print("--------------------------------------------------")
print("CA NETSCIENCE KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_kendalltau(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_kendalltau(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_spearman(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("CA NETSCIENCE SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_spearman(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_kcores_spearman(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_kendalltau(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("RESULTS (Degree Correlations)")
print("--------------------------------------------------")
print("CA NETSCIENCE KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_kendalltau(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_kendalltau(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_spearman(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("CA NETSCIENCE SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_spearman(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_degrees_spearman(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_kendalltau(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("RESULTS (Hybrid Correlations)")
print("--------------------------------------------------")
print("CA NETSCIENCE KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_kendalltau(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_kendalltau(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK KENDALL TAU")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_spearman(ca_netscience(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("CA NETSCIENCE SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_spearman(soc_wiki_vote(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("WIKIPEDIA VOTING SPEARMAN")
print("Tau:", tau)
print("p-value:", p)

tau, p=test_graph_hybrid_spearman(power_eris1176(), 0.1/15, 0.1)
print("--------------------------------------------------")
print("POWER NETWORK SPEARMAN")
print("Tau:", tau)
print("p-value:", p)
