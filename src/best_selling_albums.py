import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.cm as cm
import numpy as np

# Load and clean data
df = pd.read_csv('..\\data\\Top_10_Albums_By_Year.csv')
pruned_df = df.drop(['Artist', 'Ranking', 'Tracks', 'Album Length'], axis=1)
cleaned_df = pruned_df.drop(pruned_df[pruned_df['Year'] < 2016].index)
cleaned_df['Genre'] = cleaned_df['Genre'].str.strip().str.title()
for tuple in cleaned_df.itertuples():
    print(tuple[1], tuple[2], tuple[3], tuple[4])

# Ensure Worldwide Sales (Est.) is numeric and remove commas if present
cleaned_df['Worldwide Sales (Est.)'] = cleaned_df['Worldwide Sales (Est.)'].replace({',': ''}, regex=True).astype(float)

# Create output directory for plots
output_dir = os.path.join('..', 'graphs', 'best_selling_albums')
os.makedirs(output_dir, exist_ok=True)

# Error Bar Plot: Show the mean and variability (e.g., standard error or 95% confidence intervals) of the numerical variable across each category.
plt.figure(figsize=(10,6))
sns.barplot(
    data=cleaned_df,
    x='Genre',
    y='Worldwide Sales (Est.)',
    ci=95,
    capsize=0.2,
    errwidth=2,
)
plt.title('Error Bar Plot: Mean and 95% CI of Worldwide Sales by Genre (2016 and later)')
plt.ylabel('Worldwide Sales (Millions)')
plt.xlabel('Genre')
plt.xticks(rotation=45)
ax = plt.gca()
plt.ylim(bottom=0)  # Ensure y-axis starts at zero
# Show only 10 evenly spaced y-tick labels for reference
labels = ax.get_yticklabels()
num_labels = len(labels)
num_ticks = 10
show_indices = [round(i * (num_labels - 1) / (num_ticks - 1)) for i in range(num_ticks)]
for i, label in enumerate(labels):
    if i not in show_indices:
        label.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'error_bar_plot_mean_95ci_worldwide_sales_by_genre.png'))
plt.close()

# Barcode Chart: True barcode (rug) plot with vertical lines for each album sale, grouped by genre, each genre a different color
plt.figure(figsize=(12, 8))
genres = cleaned_df['Genre'].unique()
genres.sort()
colors = cm.get_cmap('tab20', len(genres))  # Use a categorical colormap
for i, genre in enumerate(genres):
    sales = cleaned_df.loc[cleaned_df['Genre'] == genre, 'Worldwide Sales (Est.)'] / 1_000_000  # Convert to millions for this plot only
    plt.vlines(sales, i - 0.4, i + 0.4, color=colors(i), alpha=0.7, linewidth=2)
plt.yticks(range(len(genres)), genres)
plt.xlabel('Worldwide Sales (Millions)')
plt.xlim(left=(cleaned_df['Worldwide Sales (Est.)'].min() / 1_000_000), right=(cleaned_df['Worldwide Sales (Est.)'].max() / 1_000_000) + 0.2)  # Autoscale in millions
plt.title('Barcode Chart: Individual Album Sales by Genre (2016 and later)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'barcode_chart_vertical_lines_album_sales_by_genre.png'))
plt.close()

# Histogram: Plot the distribution of the numerical variable, grouped by the categorical variable (using hue or facet).
plt.figure(figsize=(12,8))
sales_millions = cleaned_df['Worldwide Sales (Est.)'] / 1_000_000
min_bin = int(sales_millions.min())
max_bin = int(sales_millions.max()) + 1
bins = np.arange(min_bin, max_bin + 1, 1)
sns.histplot(
    data=cleaned_df.assign(**{'Worldwide Sales (Millions)': sales_millions}),
    x='Worldwide Sales (Millions)',
    hue='Genre',
    multiple='stack',
    bins=bins
)
plt.title('Histogram: Distribution of Album Sales by Genre (2016 and later)')
plt.xlabel('Worldwide Sales (Millions)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_distribution_album_sales_by_genre.png'))
plt.close()