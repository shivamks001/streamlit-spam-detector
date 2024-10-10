import pandas as pd

# Load the dataset
data = pd.read_csv('dataset.csv')

# Drop the 'Unnamed: 0' column
data = data.drop(columns=['Unnamed: 0'])

# Check the remaining columns
print(data.columns)

# Rename columns for clarity if necessary
# (This step is optional if you want more meaningful names)
data = data.rename(columns={"label": "v1", "text": "v2"})

# Preview the data to ensure it's correct
print(data.head())
