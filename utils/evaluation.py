import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(true_prices, predicted_prices):
    """
    Evaluate the performance of the model.

    Parameters:
        true_prices (array-like): Actual prices.
        predicted_prices (array-like): Predicted prices.

    Prints:
        - Mean Squared Error (MSE)
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
    """
    mse = mean_squared_error(true_prices, predicted_prices)
    mae = mean_absolute_error(true_prices, predicted_prices)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def plot_predictions (dates, true_prices, predicted_prices):
    """
    Plot actual vs. predicted prices.

    Parameters:
        dates (pandas.DatetimeIndex): Dates for the x-axis.
        true_prices (array-like): Actual prices for the y-axis.
        predicted_prices (array-like): Predicted prices for the y-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_prices, label="Actual Prices", color="blue")
    plt.plot(dates, predicted_prices, label="Predicted Prices", color="red", linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
