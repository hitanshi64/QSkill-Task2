import pandas as pd
import numpy as np

# ============================================================
# 1. LOAD KAGGLE DATASET
# ============================================================

print("=" * 60)
print("1. LOADING KAGGLE HOUSE PRICE DATASET")
print("=" * 60)

data = pd.read_csv("train.csv")

print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\nDataset loaded successfully!")


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("2. DATA PREPROCESSING")
print("=" * 60)

# Select important features from the Kaggle dataset
features = [
    "GrLivArea",       # Living area in square feet
    "BedroomAbvGr",    # Number of bedrooms
    "FullBath",        # Number of bathrooms
    "TotRmsAbvGrd",    # Total rooms
    "YearBuilt",       # Year house was built
    "Neighborhood",    # Location
    "OverallQual",     # Overall quality
    "GarageArea",      # Garage area
    "LotArea"          # Lot area
]

# Keep only required columns
data = data[features + ["SalePrice"]].copy()

# Calculate house age
data["Age"] = 2026 - data["YearBuilt"]

# Remove YearBuilt after calculating age
data.drop("YearBuilt", axis=1, inplace=True)

# Handle missing numerical values
data["GarageArea"] = data["GarageArea"].fillna(
    data["GarageArea"].median()
)

# Handle missing categorical values
data["Neighborhood"] = data["Neighborhood"].fillna(
    data["Neighborhood"].mode()[0]
)

# Convert Neighborhood into numerical columns
data = pd.get_dummies(
    data,
    columns=["Neighborhood"],
    drop_first=True
)

# Separate input features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

print("\nSelected Features:")
print("Living Area, Bedrooms, Bathrooms, Rooms, Age,")
print("Location, Quality, Garage Area and Lot Area")

print("\nPreprocessing completed successfully!")
print("Number of input features:", X.shape[1])
print("Target variable: SalePrice")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("3. TRAIN-TEST SPLIT")
print("=" * 60)

from sklearn.model_selection import train_test_split

# 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTrain-Test Split completed successfully!")


# ============================================================
# 4. LINEAR REGRESSION MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("4. LINEAR REGRESSION MODEL TRAINING")
print("=" * 60)

from sklearn.linear_model import LinearRegression

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("5. MODEL EVALUATION")
print("=" * 60)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Predict prices for test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# ============================================================
# 6. ACTUAL VS PREDICTED PRICES
# ============================================================

print("\n" + "=" * 60)
print("6. ACTUAL VS PREDICTED HOUSE PRICES")
print("=" * 60)

# Display first 10 predictions
results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": np.round(y_pred[:10], 2)
})

print("\n")
print(results.to_string(index=False))


# ============================================================
# 7. NEW HOUSE PRICE PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("7. NEW HOUSE PRICE PREDICTION")
print("=" * 60)

# Example of a new house
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2],
    "TotRmsAbvGrd": [7],
    "Age": [10],
    "Neighborhood": ["CollgCr"],
    "OverallQual": [7],
    "GarageArea": [500],
    "LotArea": [8000]
})

# Convert Neighborhood into numerical format
new_house = pd.get_dummies(
    new_house,
    columns=["Neighborhood"],
    drop_first=True
)

# Make columns same as training data
new_house = new_house.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict price
new_price = model.predict(new_house)

print("\nNew House Details:")
print("Living Area:", 2000, "sq ft")
print("Bedrooms:", 3)
print("Bathrooms:", 2)
print("Total Rooms:", 7)
print("House Age:", 10, "years")
print("Neighborhood: CollgCr")
print("Overall Quality:", 7)
print("Garage Area:", 500, "sq ft")
print("Lot Area:", 8000, "sq ft")

print("\nPredicted House Price:")
print("$", round(new_price[0], 2))


