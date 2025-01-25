import sys
import os 

# Set up the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from feature_engineering import add_technical_indicators, add_lagged_features
from data_loader import get_historical_data
from model_training import train_lstm_model 
from scaling import scale_data, inverse_scale
from evaluation import evaluate_model, plot_predictions

from model_training import prepare_lstm_data, train_lstm_model

def main():
    try:
        # Step 1: Validate user inputs
        ticker = input("Enter the stock ticker (e.g., NVDA): ").strip().upper()
        valid_periods = ['1y', '6mo', '3mo']
        period = input(f"Enter the historical period ({', '.join(valid_periods)}): ").strip()
        while period not in valid_periods:
            period = input(f"Invalid period. Please enter one of {', '.join(valid_periods)}: ").strip()

        # Step 2: Load historical data
        print(f"Fetching historical data for {ticker} with period '{period}'...")
        data = get_historical_data(ticker, period)
        if data is None or data.empty:
            raise ValueError(f"No data found for ticker {ticker} with period {period}. Please try again.")

        print("Data successfully loaded!")

        # Step 3: Add technical indicators and lagged features
        print("Adding technical indicators and lagged features...")
        data = add_technical_indicators(data)
        data = add_lagged_features(data, columns=['Close', 'Volume'], lags=5)

        # Step 4: Scale the data
        print("Scaling data...")
        scaled_data, scaler = scale_data(data)

        # Step 5: Prepare data for LSTM
        print("Preparing data for LSTM...")
        X_train, y_train, X_test, y_test = prepare_lstm_data(scaled_data, train_split=0.8)
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Insufficient data for training. Please adjust the period or features.")

        # Step 6: Train the LSTM model
        print("Training LSTM model...")
        model = train_lstm_model(X_train, y_train, epochs=50, batch_size=32)
        print("Model training completed!")

        # Step 7: Make predictions on the test set
        print("Making predictions...")
        predicted_prices_scaled = model.predict(X_test)
        predicted_prices = inverse_scale(predicted_prices_scaled, scaler)

        # Step 8: Evaluate and plot predictions
        print("Evaluating model...")
        actual_prices = y_test.flatten()
        evaluate_model(actual_prices, predicted_prices.flatten())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()