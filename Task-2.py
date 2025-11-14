import pandas as pd
import numpy as np

data = {
    'A': [10, np.nan, 30, 40],
    'B': [np.nan, 25, 35, np.nan],
    'C': [5, np.nan, 15, 20]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
df_filled = df.fillna(df.mean(numeric_only=True))

print("\nDataFrame after filling missing values with column means:")
print(df_filled)
