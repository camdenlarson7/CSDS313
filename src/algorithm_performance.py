import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import numpy as np

# Load data, set proper column names
cols = ['Index', 'Epoch', 'Algorithm', 'Run', 'Accuracy']
df = pd.read_csv('..\\data\\algorithm_trials.csv', names=cols, header=0)

# Drop the index column if it's just a row number
if df['Index'].is_monotonic_increasing and df['Index'].iloc[0] == 0:
    df = df.drop('Index', axis=1)

# Normalize algorithm names: strip whitespace, lower case, then title case
# (e.g., 'algorithm a', 'ALGORITHM A', ' Algorithm A ' -> 'Algorithm A')
df['Algorithm'] = df['Algorithm'].str.strip().str.lower().str.title()

# Convert Accuracy to numeric and drop rows with missing or invalid accuracy
# (or you could impute if desired)
df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
df = df.dropna(subset=['Accuracy'])

# Remove rows where accuracy is outside the valid domain [0, 1]
df = df[(df['Accuracy'] >= 0) & (df['Accuracy'] <= 1)]

# Create output directory for plots
output_dir = os.path.join('..', 'graphs', 'algorithm_performance')
os.makedirs(output_dir, exist_ok=True)

# Error Bar Plot: Mean and 95% CI of Accuracy by Algorithm
plt.figure(figsize=(8,6))
sns.barplot(
    data=df,
    x='Algorithm',
    y='Accuracy',
    ci=95,
    capsize=0.2,
    errwidth=2
)
plt.title('Error Bar Plot: Mean and 95% CI of Accuracy by Algorithm')
plt.ylabel('Accuracy')
plt.xlabel('Algorithm')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_bar_plot_mean_95ci_accuracy_by_algorithm.png'))
plt.close()

# Barcode Chart: True barcode (rug) plot with vertical lines for each accuracy, grouped by algorithm
plt.figure(figsize=(10, 6))
algos = df['Algorithm'].unique()
algos.sort()
colors = cm.get_cmap('tab10', len(algos))
for i, algo in enumerate(algos):
    accs = df.loc[df['Algorithm'] == algo, 'Accuracy']
    plt.vlines(accs, i - 0.4, i + 0.4, color=colors(i), alpha=0.7, linewidth=2)
plt.yticks(range(len(algos)), algos)
plt.xlabel('Accuracy')
plt.title('Barcode Chart: Individual Accuracy by Algorithm')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'barcode_chart_vertical_lines_accuracy_by_algorithm.png'))
plt.close()

# Histogram: Distribution of Accuracy by Algorithm
plt.figure(figsize=(10,6))
bins = np.arange(0, 1.05, 0.05)
sns.histplot(
    data=df,
    x='Accuracy',
    hue='Algorithm',
    multiple='stack',
    bins=bins
)
plt.title('Histogram: Distribution of Accuracy by Algorithm')
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_distribution_accuracy_by_algorithm.png'))
plt.close()