import pandas as pd
import numpy as np
from scipy import stats
import powerlaw
import matplotlib.pyplot as plt
import os

# --- Dataset 1: 10k_times.csv (Time) ---
df1 = pd.read_csv('../data/10k_times.csv')
data1 = pd.to_numeric(df1['Time'], errors='coerce').dropna().values
n1 = len(data1)

# --- Dataset 2: uniform_random_results.csv (RandomNumber) ---
df2 = pd.read_csv('../data/uniform_random_results.csv')
data2 = pd.to_numeric(df2['RandomNumber'], errors='coerce').dropna().values
n2 = len(data2)

# --- Dataset 3: call_details.csv (Day Mins, filtered to <= 10000) ---
df3 = pd.read_csv('../data/call_details.csv')
data3 = pd.to_numeric(df3['Day Mins'], errors='coerce').dropna()
data3 = data3[data3 <= 10000].values
n3 = len(data3)

# --- Dataset 4: BRAZIL_CITIES.csv (IBGE_RES_POP) ---
df4 = pd.read_csv('../data/BRAZIL_CITIES.csv', sep=';')
data4 = pd.to_numeric(df4['IBGE_RES_POP'], errors='coerce').dropna().values
n4 = len(data4)

def fit_all_models(data):
    # Normal
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    # Uniform
    a, b = np.min(data), np.max(data)
    # Power law (using powerlaw package)
    fit = powerlaw.Fit(data, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    # Exponential (MLE: lambda = 1/mean)
    lambd = 1 / np.mean(data)
    return (mu, sigma), (a, b), (alpha, xmin), lambd

results = []
for i, (name, data, n) in enumerate([
    ("Dataset 1", data1, n1),
    ("Dataset 2", data2, n2),
    ("Dataset 3", data3, n3),
    ("Dataset 4", data4, n4),
]):
    (mu, sigma), (a, b), (alpha, xmin), lambd = fit_all_models(data)
    results.append({
        'Dataset': name,
        '# Observations': n,
        'Normal': f"{mu:.2f}, {sigma:.2f}",
        'Uniform': f"{a:.2f}, {b:.2f}",
        'Power law': f"{alpha:.2f}, {xmin:.2f}",
        'Exponential': f"{lambd:.5f}"
    })

# Print results in tabular format
print(f"{'Model':<12}{'# Obs':<10}{'Normal':<20}{'Uniform':<20}{'Power law':<20}{'Exponential':<15}")
for r in results:
    print(f"{r['Dataset']:<12}{r['# Observations']:<10}{r['Normal']:<20}{r['Uniform']:<20}{r['Power law']:<20}{r['Exponential']:<15}")

# Helper: Generate synthetic data for each model
synthetic = {}
for idx, (name, data, n) in enumerate([
    ("Dataset 1", data1, n1),
    ("Dataset 2", data2, n2),
    ("Dataset 3", data3, n3),
    ("Dataset 4", data4, n4),
]):
    (mu, sigma), (a, b), (alpha, xmin), lambd = fit_all_models(data)
    np.random.seed(42)
    synth = {}
    # Normal
    synth['Normal'] = np.random.normal(mu, sigma, n)
    # Uniform
    synth['Uniform'] = np.random.uniform(a, b, n)
    # Power law (shifted by xmin)
    synth['Power law'] = (np.random.pareto(alpha, n) + 1) * xmin
    # Exponential
    synth['Exponential'] = np.random.exponential(1/lambd, n)
    synthetic[name] = synth

# Visualization and comparison
def plot_real_vs_synth(real, synth_dict, dataset_name, outdir):
    plt.figure(figsize=(12,8))
    for i, (model, synth) in enumerate(synth_dict.items()):
        plt.hist(synth, bins=50, alpha=0.5, label=f'{model} (synthetic)', density=True)
    plt.hist(real, bins=50, alpha=0.7, label='Real Data', color='black', density=True, histtype='step')
    plt.title(f'Real vs. Synthetic Distributions for {dataset_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'{dataset_name.lower().replace(" ", "_")}_real_vs_synth.png'))
    plt.close()

# Compare using Kolmogorov-Smirnov statistic
def compare_ks(real, synth_dict):
    scores = {}
    for model, synth in synth_dict.items():
        stat, _ = stats.ks_2samp(real, synth)
        scores[model] = stat
    return scores

# Run for each dataset
outdir = '../graphs/model_fits'
most_similar = {}
for idx, (name, data, n) in enumerate([
    ("Dataset 1", data1, n1),
    ("Dataset 2", data2, n2),
    ("Dataset 3", data3, n3),
    ("Dataset 4", data4, n4),
]):
    plot_real_vs_synth(data, synthetic[name], name, outdir)
    ks_scores = compare_ks(data, synthetic[name])
    best_model = min(ks_scores, key=ks_scores.get)
    most_similar[name] = best_model
    print(f"{name}: Most similar synthetic model by KS test: {best_model} (KS={ks_scores[best_model]:.4f})")


