import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

# Save 1000 runs of random numbers to a CSV file
with open('../data/uniform_random_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Run', 'RandomNumber'])
    for i in range(1, 50001):
        random_number = random.randint(1, 100)
        writer.writerow([i, random_number])

# Create output directory for the plot
output_dir = '../graphs/d100_rolls'
os.makedirs(output_dir, exist_ok=True)

# Plot the distribution of values between 1 and 100
results_df = pd.read_csv('../data/uniform_random_results.csv')
plt.figure(figsize=(10,6))
plt.hist(results_df['RandomNumber'], bins=range(1, 102), edgecolor='black', align='left')
plt.xlabel('Random Number (1-100)')
plt.ylabel('Frequency')
plt.title('Distribution of Uniform Random Numbers (1-100)')
plt.xticks(range(0, 101, 5))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_uniform_random_distribution.png'))
plt.close()
