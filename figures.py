import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from epidemic_models import SIR
from graph import Graph
from generate_graphs import *
from test import *
from scipy.stats import chi2, spearmanr, kendalltau

def heatmaps():
    row_labels=["Degree", "Eigenvector", "K-core", "Hybrid"]
    col_labels=["ER-τ", "ER-ρ", "CP-τ", "CP-ρ", "Geo-τ", "Geo-ρ"]
    data = [
        # ER-τ      ER-ρ      CP-τ      CP-ρ      Geo-τ     Geo-ρ
        [0.14444055987292242, 0.18208269552009537, 0.314641735319005, 0.3766289607979403, 0.16860596669745798, 0.22534749132344534],  # Degree
        [0.13358215485651356, 0.16143106170418384, 0.27540086647941103, 0.36329774749962807, 0.06494850467365434, 0.10301869265572465],  # Eigenvector
        [0.10774899768286074, 0.1280991107191252, 0.30732569214322664, 0.37343969047165876, 0.15794144083410128, 0.18195713420619397],  # K-core
        [0.13662357805262856, 0.16273752358976512, 0.29849007681899203, 0.39693634575796943, 0.07328044179285138, 0.10773931266289298]   # Hybrid
    ]
    data=np.array(data)

    plt.figure(figsize=(10, 5))

    plt.imshow(data, cmap="viridis")

    cbar = plt.colorbar()
    cbar.set_label("Correlation", rotation=-90, labelpad=15)

    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, f"{data[i, j]:.2f}",
                     ha="center", va="center", color="white")

    plt.title("Correlation Heatmap (τ and ρ) Across Metrics and Networks")
    plt.tight_layout()
    plt.savefig("correlation_synthetic.png")
    plt.clf()

    data = [
        # ER-τ      ER-ρ      CP-τ      CP-ρ      Geo-τ     Geo-ρ
        [3.042716134872578, 2.9473448397200377, 6.537176054866378, 7.064105626618678, 3.974837203711327, 5.4038112234137845],  # Degree
        [3.8359321787337577, 2.994883398519661, 6.584873182847409, 7.595505865418919, 1.328219795647139, 1.7288486981397924],  # Eigenvector
        [2.14371754537246, 2.50606519485545, 6.89900937928574, 6.3295904440333794, 2.52422111950483, 2.5984259848743823],        # K-core
        [4.043340858028323, 2.918098565696871, 8.11166483608601, 7.9694833073220135, 1.3775612888848525, 1.7628836437717001]   # Hybrid
    ]

    data=np.array(data)

    plt.figure(figsize=(10, 5))

    plt.imshow(data, cmap="viridis")

    cbar = plt.colorbar()
    cbar.set_label("Effect Size", rotation=-90, labelpad=15)

    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, f"{data[i, j]:.2f}",
                     ha="center", va="center", color="white")

    plt.title("Cohen's d Effect Size Heatmap Across Metrics and Networks")
    plt.tight_layout()
    plt.savefig("cohensd_synthetic.png")
    plt.clf()

def heatmaps_real_world():
    corr = np.array([
        [0.0618074632347477, 0.265992274519672, 0.34842187593885043,
         0.06277391737836983, 0.2838836534346215, 0.4416824517402113],

        [0.1909236136402187, 0.3609727239095073, 0.5241153244733623,
         0.18087346290856882, 0.3789004143184187, 0.5842853300213281],

        [0.15048989655176534, 0.3327071025483964, 0.5170038474619211,
         0.3089015440991835, 0.3785710793172163, 0.6104103494245247],

        [0.10296034122649003, 0.2799720871023732, 0.36332042448727075,
         0.1319948786557211, 0.39600349546454955, 0.47945816613445047]
    ])

    row_labels = [
        "Perron Eigenvector",
        "k-core Index",
        "Degree",
        "Hybrid"
    ]

    col_labels = [
        "CA-Kendall", "Wiki-Kendall", "Power-Kendall",
        "CA-Spearman", "Wiki-Spearman", "Power-Spearman"
    ]

    plt.figure(figsize=(10, 6))
    plt.imshow(corr, cmap="viridis", aspect="auto")
    plt.colorbar(label="Correlation Coefficient")

    # Tick labels
    plt.xticks(np.arange(6), col_labels, rotation=45, ha="right")
    plt.yticks(np.arange(4), row_labels)

    # Annotate each cell with its value
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr[i, j]:.2f}",
                     ha="center", va="center", color="white" if corr[i, j] > 0.35 else "black")

    plt.title("Correlation Heatmap (τ and ρ) Across Metrics and Real-World Networks")
    plt.tight_layout()
    plt.savefig("realworld_heatmaps.png")



heatmaps_real_world()


