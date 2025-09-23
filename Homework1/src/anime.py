import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import numpy as np

# Load data
df = pd.read_csv('..\\data\\Anime.csv')

# Strip leading/trailing whitespace from all string columns before filtering
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# Keep only TV series and movies released after 2015
filtered_df = df[(df['Type'].isin(['TV', 'Movie'])) & (df['Release_year'] > 2015)]

# Convert Rating column to numeric after cleaning and filtering
filtered_df['Rating'] = pd.to_numeric(filtered_df['Rating'], errors='coerce')

# Keep only relevant columns for the research question
relevant_columns = ['Name', 'Type', 'Episodes', 'Studio', 'Release_year', 'Rating']
filtered_df = filtered_df[relevant_columns]

# Create output directory for plots
output_dir = os.path.join('..', 'graphs', 'anime')
os.makedirs(output_dir, exist_ok=True)

# Error Bar Plot: Mean and 95% CI of Ratings by Type
plt.figure(figsize=(8,6))
sns.barplot(
    data=filtered_df,
    x='Type',
    y='Rating',
    ci=95,
    capsize=0.2,
    errwidth=2
)
plt.title('Error Bar Plot: Mean and 95% CI of Ratings by Anime Type (2016 and later)')
plt.ylabel('Audience Rating')
plt.xlabel('Anime Type')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_bar_plot_mean_95ci_rating_by_type.png'))
plt.close()

# Barcode Chart: True barcode (rug) plot with vertical lines for each rating, grouped by type
plt.figure(figsize=(8, 4))
types = filtered_df['Type'].unique()
types.sort()
colors = cm.get_cmap('tab10', len(types))
for i, t in enumerate(types):
    ratings = filtered_df.loc[filtered_df['Type'] == t, 'Rating']
    plt.vlines(ratings, i - 0.4, i + 0.4, color=colors(i), alpha=0.7, linewidth=2)
plt.yticks(range(len(types)), types)
plt.xlabel('Audience Rating')
plt.title('Barcode Chart: Individual Ratings by Anime Type (2016 and later)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'barcode_chart_vertical_lines_rating_by_type.png'))
plt.close()

# Histogram: Distribution of Ratings by Type
plt.figure(figsize=(10,6))
min_bin = np.floor(filtered_df['Rating'].min() * 10) / 10
max_bin = np.ceil(filtered_df['Rating'].max() * 10) / 10
bins = np.arange(min_bin, max_bin + 0.11, 0.1)
sns.histplot(
    data=filtered_df,
    x='Rating',
    hue='Type',
    multiple='stack',
    bins=bins
)
plt.title('Histogram: Distribution of Ratings by Anime Type (2016 and later)')
plt.xlabel('Audience Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_distribution_rating_by_type.png'))
plt.close()