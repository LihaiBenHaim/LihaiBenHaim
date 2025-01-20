import tensorflow as tf
from tensorflow.keras import layers, models

def build_anomaly_detector(input_dim):
    """
    Autoencoder-based anomaly detection for high-throughput data streams.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    detector = build_anomaly_detector(128)
    detector.summary()