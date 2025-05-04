import numpy as np
import pandas as pd
import argparse
from typing import List, Dict
import time
import sys
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self, units: int, activation: str = 'sigmoid'):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input_shape = None
        
    def initialize(self, input_shape: int):
        self.input_shape = input_shape
        limit = np.sqrt(6 / self.input_shape)
        self.weights = np.random.uniform(-limit, limit, (self.input_shape, self.units))
        self.bias = np.zeros((1, self.units))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'softmax':
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:  # Default to sigmoid
            self.output = 1 / (1 + np.exp(-self.z))
            
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.activation == 'softmax':
            grad_z = grad_output
        else:  # sigmoid
            grad_activation = self.output * (1 - self.output)
            grad_z = grad_output * grad_activation
            
        batch_size = self.inputs.shape[0]
        self.grad_weights = np.dot(self.inputs.T, grad_z) / batch_size
        self.grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def update_params(self, learning_rate: float):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias

class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []
        self.metrics_history = {
            'loss': [],
            'val_loss': []
        }

    def add(self, layer: DenseLayer):
        self.layers.append(layer)
    
    def build(self, input_shape: int):
        prev_units = input_shape
        for layer in self.layers:
            layer.initialize(prev_units)
            prev_units = layer.units
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        outputs = X
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        batch_size = y_true.shape[0]
        
        # Use cross-entropy loss when using softmax activation
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Categorical cross-entropy loss with one-hot encoded labels
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        # Gradient of cross-entropy loss with softmax
        grad_output = y_pred - y_true
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
            
        return loss
    
    def update_params(self, learning_rate: float):
        for layer in self.layers:
            layer.update_params(learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 10, learning_rate: float = 0.01, 
              batch_size: int = 32) -> Dict[str, List[float]]:
        
        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass and update parameters
                batch_loss = self.backward(y_batch, y_pred)
                self.update_params(learning_rate)
                
                epoch_loss += batch_loss
            
            # Calculate average loss for the epoch
            epoch_loss /= num_batches
            
            # Validation
            y_val_pred = self.forward(X_val)
            
            # Use the same categorical cross-entropy for validation
            epsilon = 1e-15
            y_val_pred = np.clip(y_val_pred, epsilon, 1 - epsilon)
            val_loss = -np.sum(y_val * np.log(y_val_pred)) / y_val.shape[0]
            
            # Store metrics
            self.metrics_history['loss'].append(epoch_loss)
            self.metrics_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f'epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}')
        
        return self.metrics_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Get softmax probabilities
        probs = self.forward(X)
        
        # For binary classification, extract probability of positive class (class 1)
        if probs.shape[1] == 2:  # If using softmax with 2 classes
            return probs[:, 1].reshape(-1, 1)  # Return positive class probability
        else:
            return probs
    
    def save(self, filepath: str):
        model_data = []
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'bias': layer.bias,
                'activation': layer.activation,
                'units': layer.units
            }
            model_data.append(layer_data)
        
        np.save(filepath, model_data)
        print(f"> saving model '{filepath}' to disk...")
    
    def load(self, filepath: str):
        model_data = np.load(filepath, allow_pickle=True)
        
        if len(model_data) != len(self.layers):
            raise ValueError(f"Model architecture mismatch: saved model has {len(model_data)} layers, but current model has {len(self.layers)} layers")
        
        for i, layer_data in enumerate(model_data):
            self.layers[i].weights = layer_data['weights']
            self.layers[i].bias = layer_data['bias']
            self.layers[i].activation = layer_data['activation']
            self.layers[i].units = layer_data['units']
            
            if i > 0:  # Skip input layer
                self.layers[i].input_shape = self.layers[i-1].units
            
        print(f"> loaded model from '{filepath}'")

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate binary cross-entropy error as specified in the project"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    N = y_true.shape[0]
    bce = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / N
    
    return bce

def plot_learning_curves(history, save_path=None):
    """Plot learning curves for loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='training loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def preprocess_data(data):
    """Preprocess the dataset for training/prediction"""
    # Extract features and labels
    X = data.iloc[:, :-1].values
    
    # If the last column contains 'M' and 'B', convert to 1 and 0
    last_col = data.iloc[:, -1]
    if last_col.dtype == 'object':
        y = np.array([[1 if label == 'M' else 0] for label in last_col])
    else:
        y = data.iloc[:, -1:].values
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Multilayer perceptron for breast cancer classification')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help='Mode to run: train or predict')
    # Set default for --data depending on mode
    default_data = 'datasets/Validation.csv' if '--mode' in sys.argv and 'predict' in sys.argv else 'datasets/data.csv'
    parser.add_argument('--data', type=str, default=default_data, help='Path to data CSV file for prediction')
    parser.add_argument('--train', type=str, default='datasets/Training.csv', help='Path to training CSV file')
    parser.add_argument('--valid', type=str, default='datasets/Validation.csv', help='Path to validation CSV file')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help='Number of units in each hidden layer')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='model.npy', help='Path to save/load model')
    
    args = parser.parse_args()
    
    # TRAIN MODE
    if args.mode == 'train':
        if not args.train or not args.valid:
            parser.error("--train mode requires --train and --valid")
        
        train_data = pd.read_csv(args.train)
        valid_data = pd.read_csv(args.valid)
        
        X_train, y_train = preprocess_data(train_data)
        X_valid, y_valid = preprocess_data(valid_data)
        
        # Convert to one-hot encoding for categorical cross-entropy
        y_train_one_hot = np.zeros((y_train.shape[0], 2))
        y_train_one_hot[np.arange(y_train.shape[0]), y_train.flatten().astype(int)] = 1
        
        y_valid_one_hot = np.zeros((y_valid.shape[0], 2))
        y_valid_one_hot[np.arange(y_valid.shape[0]), y_valid.flatten().astype(int)] = 1
        
        print(f'x_train shape : {X_train.shape}')
        print(f'y_train shape : {y_train_one_hot.shape}')
        print(f'x_valid shape : {X_valid.shape}')
        print(f'y_valid shape : {y_valid_one_hot.shape}')
        
        model = MultiLayerPerceptron()
        
        input_shape = X_train.shape[1]
        for units in args.layer:
            model.add(DenseLayer(units, activation='sigmoid'))
        model.add(DenseLayer(2, activation='softmax'))

        model.build(input_shape)
        
        # Train the model
        history = model.train(
            X_train, y_train_one_hot,
            X_valid, y_valid_one_hot,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        # Save the model
        model.save(args.model)
        
        # Plot learning curves
        plot_learning_curves(history)
    
    # PREDICT MODE
    elif args.mode == 'predict':
        if not args.data or not args.model:
            parser.error("--predict mode requires --data and --model")
        
        data = pd.read_csv(args.data)
        X, y = preprocess_data(data)
        
        model = MultiLayerPerceptron()
        
        # Create model with same architecture as training
        input_shape = X.shape[1]
        for units in args.layer:
            model.add(DenseLayer(units, activation='sigmoid'))
        model.add(DenseLayer(2, activation='softmax'))
        
        # Initialize and load weights
        model.build(input_shape)
        model.load(args.model)
        
        # Get predictions (probabilities of positive class)
        positive_probs = model.predict(X)
        
        # Calculate binary cross-entropy as required by the project
        bce = binary_cross_entropy(y, positive_probs)
        print(f"Binary Cross-Entropy: {bce:.4f}")
        
        # Calculate accuracy
        predicted_classes = (positive_probs >= 0.5).astype(int)
        accuracy = np.mean(predicted_classes == y)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Display some predictions
        print("\nSample predictions (First 5):")
        print("True\tPred\tProbability")
        for i in range(min(5, len(y))):
            print(f"{y[i, 0]}\t{predicted_classes[i, 0]}\t{positive_probs[i, 0]:.4f}")

if __name__ == "__main__":
    main()