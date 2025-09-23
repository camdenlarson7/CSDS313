import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Brazil cities data (semicolon separator)
cities_df = pd.read_csv('../data/BRAZIL_CITIES.csv', sep=';')

# Create a dataframe with just city name and population
city_pop_df = cities_df[['CITY', 'IBGE_RES_POP']].copy()
city_pop_df = city_pop_df.dropna(subset=['IBGE_RES_POP'])
city_pop_df['IBGE_RES_POP'] = pd.to_numeric(city_pop_df['IBGE_RES_POP'], errors='coerce')
city_pop_df = city_pop_df.dropna(subset=['IBGE_RES_POP'])

# Create output directory for the plot
output_dir = '../graphs/city_populations'
os.makedirs(output_dir, exist_ok=True)

# Use logarithmic bins for the histogram
min_pop = city_pop_df['IBGE_RES_POP'].min()
max_pop = city_pop_df['IBGE_RES_POP'].max()
bins = np.logspace(np.log10(min_pop), np.log10(max_pop), 50)

plt.figure(figsize=(10,6))
plt.hist(city_pop_df['IBGE_RES_POP'], bins=bins, edgecolor='black', log=True)
plt.xscale('log')
plt.xlabel('City Population (log scale)')
plt.ylabel('Number of Cities (log scale)')
plt.title('Distribution of City Populations in Brazil (Log Bins)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_brazil_city_populations.png'))
plt.close()
