import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("D:\datasets\BHP.csv")

# Step 2: Explore the data
print(df.head())
print(df.info())
print(df.isnull().sum())

# Step 3: Data Cleaning
# Drop rows with missing target prices
df = df.dropna(subset=['price'])

# Fill missing values for other columns
df['area_type'].fillna(df['area_type'].mode()[0], inplace=True)
df['location'].fillna(df['location'].mode()[0], inplace=True)
df['size'].fillna(df['size'].mode()[0], inplace=True)
df['bath'].fillna(df['bath'].mean(), inplace=True)
df['balcony'].fillna(df['balcony'].mean(), inplace=True)

# Check for missing values again
print(df.isnull().sum())

# Step 4: Feature Selection and Encoding
features = ['total_sqft', 'bath', 'balcony', 'area_type', 'location']
X = df[features]
y = df['price']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Display the first few rows after encoding
print(X.head())

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Step 9: Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.show()





