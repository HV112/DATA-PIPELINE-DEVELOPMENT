import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------
# EXTRACT: Load data
# ----------------------------
data = pd.read_csv("data.csv")
print("Original Data:")
print(data)

# ----------------------------
# PREPROCESS: Handle missing values
# ----------------------------
# Fill numeric columns with mean
numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Fill categorical columns with mode
categorical_columns = data.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nAfter Preprocessing:")
print(data)

# ----------------------------
# TRANSFORM: Encoding & Scaling
# ----------------------------
data_encoded = pd.get_dummies(data, drop_first=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_encoded)

processed_data = pd.DataFrame(scaled_data, columns=data_encoded.columns)

print("\nAfter Transformation:")
print(processed_data)

# ----------------------------
# LOAD: Save processed data
# ----------------------------
processed_data.to_csv("processed_data.csv", index=False)
print("\nProcessed data saved as processed_data.csv")
