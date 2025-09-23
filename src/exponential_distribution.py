import pandas as pd
import matplotlib.pyplot as plt
import os

# Load call details data
call_df = pd.read_csv('../data/call_details.csv')

# Clean and convert 'Day Mins' to numeric, drop missing values
call_df['Day Mins'] = pd.to_numeric(call_df['Day Mins'], errors='coerce')
call_df = call_df.dropna(subset=['Day Mins'])

# Remove all rows where 'Day Mins' > 10000
filtered_df = call_df[call_df['Day Mins'] <= 10000]

print('Day Mins summary:')
print(filtered_df['Day Mins'].describe())
print('Max:', filtered_df['Day Mins'].max())
print('Min:', filtered_df['Day Mins'].min())
print('Sample values:', filtered_df['Day Mins'].head(10).tolist())

# Create output directory for the plot
output_dir = '../graphs/exponential_distribution'
os.makedirs(output_dir, exist_ok=True)

# Plot histogram of total call time (Day Mins) with rescaled x-axis
plt.figure(figsize=(10,6))
plt.hist(filtered_df['Day Mins'], bins=50, edgecolor='black')
plt.xlabel('Total Call Time (Day Mins)')
plt.ylabel('Number of Customers')
plt.title('Distribution of Total Call Time (Day Mins)\n(>10,000 Mins Removed)')
plt.xlim(filtered_df['Day Mins'].min(), filtered_df['Day Mins'].max())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_day_mins_no_outliers.png'))
plt.close()
