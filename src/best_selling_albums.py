import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('..\data\Top_10_Albums_By_Year.csv')

# Replace 'Genre' with the actual column name for genre in your CSV
genre_counts = df['Genre'].value_counts()

plt.figure(figsize=(10,6))
genre_counts.plot(kind='bar')
plt.title('Number of Top Selling Albums by Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Albums')
plt.tight_layout()
plt.show()