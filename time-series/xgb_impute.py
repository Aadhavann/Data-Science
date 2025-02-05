import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier

# Example dataset
data = pd.read_csv("path/to/data.csv")

# Encode categorical variables while keeping NaN as NaN
def encode_categorical_with_nan(data, column):
    categories = data[column].dropna().unique()
    category_mapping = {category: code for code, category in enumerate(categories)}
    data[column] = data[column].map(category_mapping).where(data[column].notna(), np.nan)
    return data

data = encode_categorical_with_nan(data, 'Education Level')

# Imputation function using XGBoost
def impute_with_xgboost(data):
    data_copy = data.copy()

    for column in data_copy.columns:
        if data[column].isnull().sum() > 0:
            print(f"Imputing column: {column}")
            
            non_missing = data_copy.loc[data[column].notna()]
            missing = data_copy.loc[data[column].isna()]
            
            X_train = non_missing.drop(columns=[column])
            y_train = non_missing[column]
            X_missing = missing.drop(columns=[column])
            
            if data[column].dtype == np.float64 or data[column].dtype == np.int64:
                model = XGBRegressor(n_estimators=100, random_state=42)
            else:
                model = XGBClassifier(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_missing)
            data_copy.loc[data[column].isna(), column] = predictions
    
    return data_copy

# Apply the imputation
imputed_data = impute_with_xgboost(data)
print(imputed_data)
