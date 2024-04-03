# Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(filepath):
    """
    Load the data and return the DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        exit()

def display_data_details(df):
    """
    Display initial details of the dataframe.
    """
    print(df.head(100))
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nData types for each column:")
    print(df.dtypes)

def linear_regression_fit(X, y):
    """
    Perform linear regression and return model, predictions, mse, and r2.
    """
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, y_pred, mse, r2

def plot_fit(X, y, y_pred, xlabel, ylabel):
    """
    Plot the data and the linear regression fit.
    """
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.title("Linear Regression Fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Load data
btc_df = load_data("/Users/brocktbennett/GitHub/Project Data/mini_project/rxdata_401104_MiniProject_1_BTC.csv")

# Display initial data details
display_data_details(btc_df)

# Set feature and target columns
X_open = btc_df[['open']]
y = btc_df['close']

# Perform linear regression using 'open' column
model, y_pred, mse, r2 = linear_regression_fit(X_open, y)
print(f"\nMean Squared Error: {mse}")
print(f"R^2 (coefficient of determination): {r2}")

# Plot the results
plot_fit(X_open, y, y_pred, "open", "close")

# Convert 'Snapshotdatetime' to pandas datetime format and calculate 'Days' column
btc_df['Snapshotdatetime'] = pd.to_datetime(btc_df['Snapshotdatetime'])
btc_df['Days'] = (btc_df['Snapshotdatetime'] - btc_df['Snapshotdatetime'].min()).dt.days

# Set new feature column
X_days = btc_df[['Days']]

# Perform linear regression using 'Days' column
model, y_pred, mse, r2 = linear_regression_fit(X_days, y)

# Plot the results
plot_fit(X_days, y, y_pred, "Days since start date", "close")
