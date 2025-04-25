# -*- coding: utf-8 -*-
"""
LSTM model for blink detection using TensorFlow/Keras.
"""
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class BlinkLSTMModel:
    """
    LSTM model for sequence-based blink detection.
    Uses temporal patterns in eye aspect ratio (EAR) to detect blinks.
    """
    def __init__(self, sequence_length=10, model_path=None):
        self.sequence_length = sequence_length
        self.model = None
        self.model_path = model_path
        
        # Load existing model or create new one
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._create_model()
            
        logger.info("LSTM model initialized")

    def _create_model(self):
        """Create a new LSTM model for blink detection."""
        logger.info("Creating new LSTM model")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("LSTM model created")
        return model

    def _load_model(self, model_path):
        """Load an existing LSTM model from file."""
        try:
            logger.info(f"Loading LSTM model from {model_path}")
            self.model = load_model(model_path)
            logger.info("LSTM model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._create_model()
            return False

    def prepare_sequences(self, ear_values, blink_labels=None):
        """
        Prepare sequences for training or prediction.
        
        Args:
            ear_values: List of EAR values
            blink_labels: Optional list of blink labels (1 for blink, 0 for no blink)
            
        Returns:
            X: Sequence data with shape (n_samples, sequence_length, 1)
            y: Labels with shape (n_samples, 1) if blink_labels provided, otherwise None
        """
        if len(ear_values) < self.sequence_length:
            logger.warning(f"Not enough EAR values to create sequences (have {len(ear_values)}, need {self.sequence_length})")
            return None, None
        
        # Create sequences
        sequences = []
        labels = []
        
        for i in range(len(ear_values) - self.sequence_length + 1):
            seq = ear_values[i:i + self.sequence_length]
            sequences.append(seq)
            
            if blink_labels is not None:
                # For training: label is for the last element in the sequence
                labels.append(blink_labels[i + self.sequence_length - 1])
        
        # Convert to numpy arrays and reshape for LSTM
        X = np.array(sequences).reshape(-1, self.sequence_length, 1)
        
        if blink_labels is not None:
            y = np.array(labels).reshape(-1, 1)
            return X, y
        else:
            return X, None

    def train(self, X_train, y_train, validation_split=0.2, epochs=20, batch_size=32):
        """
        Train the LSTM model with prepared sequence data.
        
        Args:
            X_train: Numpy array with shape (n_samples, sequence_length, 1)
            y_train: Numpy array with shape (n_samples, 1)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            History object from model training
        """
        if X_train is None or y_train is None:
            logger.error("Cannot train with None data")
            return None
            
        logger.info(f"Training LSTM model with {len(X_train)} sequences")
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ]
        
        if self.model_path:
            callbacks.append(
                ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
            )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Model training completed")
        return history

    def predict(self, X):
        """
        Predict blink probability for sequences.
        
        Args:
            X: Numpy array with shape (n_samples, sequence_length, 1)
            
        Returns:
            Numpy array of predictions with shape (n_samples, 1)
        """
        if X is None:
            logger.error("Cannot predict with None data")
            return None
            
        if self.model is None:
            logger.error("Model not initialized")
            return None
            
        return self.model.predict(X, verbose=0)

    def save_model(self, model_path=None):
        """Save the trained model to disk."""
        if model_path is None:
            model_path = self.model_path
            
        if model_path is None:
            logger.error("No model path specified for saving")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Update model path
            self.model_path = model_path
            
            # Convert to TensorFlow Lite format
            self._convert_to_tflite(model_path)
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def _convert_to_tflite(self, model_path):
        """Convert the model to TensorFlow Lite format for faster inference."""
        try:
            # Get TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Set optimization flags
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the TFLite model
            tflite_path = f"{model_path}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            logger.info(f"TFLite model saved to {tflite_path}")
            return True
        except Exception as e:
            logger.error(f"Error converting to TFLite: {str(e)}")
            return False

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        
        Args:
            X_test: Numpy array with shape (n_samples, sequence_length, 1)
            y_test: Numpy array with shape (n_samples, 1)
            
        Returns:
            Dictionary with loss and accuracy
        """
        if X_test is None or y_test is None:
            logger.error("Cannot evaluate with None data")
            return None
            
        if self.model is None:
            logger.error("Model not initialized")
            return None
            
        results = self.model.evaluate(X_test, y_test, verbose=0)
        return {"loss": results[0], "accuracy": results[1]}
