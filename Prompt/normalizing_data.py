import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the CSV file
df = pd.read_csv('/home/tnadolsk/Mus2Vid-code/Prompt_Generation/data.csv', delimiter='\t')

df.columns = df.columns.str.strip()

# Specify the column names for normalization
columns_to_normalize = ['Arousal', 'Valence']

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(-1, 1))

# Normalize the selected columns
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Access the normalized data
df.to_csv('/home/tnadolsk/Mus2Vid-code/Prompt_Generation/normalized_data.csv', index=False)

print("Data saved to new CSV file successfully.")
