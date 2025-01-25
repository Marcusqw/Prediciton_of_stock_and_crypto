from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout 
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizer_v2 import adam as Adam
print("Imports are working correctly!")

def prepare_lstm_data(data, target_column='Close', sequence_length=60, train_split=0.8):
    """
    Prepare data for LSTM training.

    Parameters:
        data (pd.DataFrame): DataFrame containing the features and target column.
        target_column (str): The name of the column to predict.
        sequence_length (int): The length of the input sequences.
        train_split (float): The proportion of data to use for training.

    Returns:
        np.array: X_train (input features for training)
        np.array: y_train (target values for training)
        np.array: X_test (input features for testing)
        np.array: y_test (target values for testing)
    """
    # Extract the target column and other features
    target = data[target_column].values
    features = data.values

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(target)):
        X.append(features[i-sequence_length:i])  # Sequence of features
        y.append(target[i])  # Target value

    X, y = np.array(X), np.array(y)

    # Split into training and testing sets
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test

def train_lstm_model(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
    """
    Train an LSTM model for time series prediction.

    Parameters:
        X_train (np.array): Input features for training.
        y_train (np.array): Target values for training.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        validation_split (float): Proportion of training data to use for validation.

    Returns:
        keras.Model: Trained LSTM model.
    """
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Third LSTM layer
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))

    # Dense layer
    model.add(Dense(1))

    # Compile model
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model with validation split
    model.fit(X_train, y_train, 
              validation_split=validation_split, 
              epochs=epochs, 
              batch_size=batch_size, 
              callbacks=[early_stop])

    return model
