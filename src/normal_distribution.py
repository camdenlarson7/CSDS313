import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
# Use 10k_times.csv and the 'Time' column (in seconds)
df = pd.read_csv('..\\data\\10k_times.csv')

df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
df = df.dropna(subset=['Time'])

# Create output directory for plots
output_dir = os.path.join('..', 'graphs', 'running_times')
os.makedirs(output_dir, exist_ok=True)

min_time = int(df['Time'].min())
max_time = int(df['Time'].max()) + 1

# Generate 10 histograms with bin sizes from 3 to 30 seconds
for i, bin_size in enumerate(range(3, 33, 3), start=1):
    bins = np.arange(min_time, max_time + bin_size, bin_size)
    plt.figure(figsize=(10,6))
    n, bins_, patches = plt.hist(df['Time'], bins=bins, edgecolor='black')
    plt.xticks(
        bins,
        [f"{int(b//60)}:{int(b%60):02d}" for b in bins],
        rotation=45
    )
    plt.xlabel('10K Time (minutes:seconds)')
    plt.ylabel('Number of Occurrences')
    plt.title(f'Histogram of 10K Times (Bin Size: {bin_size} seconds)')
    plt.tight_layout()
    filename = f"histogram_10k_time_bin{bin_size}s.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Remove top 5 and bottom 5 outliers
sorted_times = df['Time'].sort_values().reset_index(drop=True)
filtered_times = sorted_times[5:-5]

min_time_f = int(filtered_times.min())
max_time_f = int(filtered_times.max()) + 1

# Generate 10 histograms with bin sizes from 3 to 30 seconds (filtered data)
for i, bin_size in enumerate(range(3, 33, 3), start=1):
    bins = np.arange(min_time_f, max_time_f + bin_size, bin_size)
    plt.figure(figsize=(10,6))
    n, bins_, patches = plt.hist(filtered_times, bins=bins, edgecolor='black')
    plt.xticks(
        bins,
        [f"{int(b//60)}:{int(b%60):02d}" for b in bins],
        rotation=45
    )
    plt.xlabel('10K Time (minutes:seconds)')
    plt.ylabel('Number of Occurrences')
    plt.title(f'Histogram of 10K Times (Bin Size: {bin_size} seconds, 5 Top/Bottom Outliers Removed)')
    plt.tight_layout()
    filename = f"histogram_10k_time_bin{bin_size}s_no_outliers.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
