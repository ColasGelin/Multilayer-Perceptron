import numpy as np
import pandas as pd
import argparse
from typing import List, Dict
import time

class DenseLayer:
    def __init__(self, units: int, activation: str = 'sigmoid'):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input_shape = None
        
    def initialize(self, input_shape: int):
        self.input_shape = input_shape
        
        self.weights = np.random.randn(self.input_shape, self.units) * 0.01
        self.bias = np.zeros((1, self.units))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function (sigmoid or softmax only)
        if self.activation == 'softmax':
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:  # Default to sigmoid
            self.output = 1 / (1 + np.exp(-self.z))
            
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Calculate gradient based on activation function
        if self.activation == 'softmax':
            grad_z = grad_output
        else:  # sigmoid
            grad_activation = self.output * (1 - self.output)
            grad_z = grad_output * grad_activation
            
        # Calculate gradients w.r.t. weights, bias, and inputs
        batch_size = self.inputs.shape[0]
        self.grad_weights = np.dot(self.inputs.T, grad_z) / batch_size
        self.grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        
        # Gradient w.r.t inputs for backpropagation to previous layer
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
        
        # Initialize each layer with correct shape
        for layer in self.layers:
            layer.initialize(prev_units)
            prev_units = layer.units
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        outputs = X
        
        # Forward pass through each layer
        for layer in self.layers:
            outputs = layer.forward(outputs)
            
        return outputs
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        batch_size = y_true.shape[0]
        
        # Using MSE loss for simplicity
        loss = np.mean(np.square(y_true - y_pred))
        
        # Gradient of MSE loss
        grad_output = -2 * (y_true - y_pred) / batch_size
        
        # Backpropagate through layers in reverse order
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
            val_loss = np.mean(np.square(y_val - y_val_pred))
            
            # Store metrics
            self.metrics_history['loss'].append(epoch_loss)
            self.metrics_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f'epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}')
        
        return self.metrics_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
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

def main():
    parser = argparse.ArgumentParser(description='Train a multilayer perceptron')
    parser.add_argument('--train', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--valid', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help='Number of units in each hidden layer')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_model', type=str, default='./saved_model.npy', help='Path to save the model')
    
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train)
    valid_data = pd.read_csv(args.valid)
    
    # Extract features and labels
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1:].values
    X_valid = valid_data.iloc[:, :-1].values
    y_valid = valid_data.iloc[:, -1:].values
    
    print(f'x_train shape : {X_train.shape}')
    print(f'x_valid shape : {X_valid.shape}')
    
    model = MultiLayerPerceptron()
    
    input_shape = X_train.shape[1]
    for units in args.layer:
        model.add(DenseLayer(units, activation='sigmoid'))
    model.add(DenseLayer(1, activation='sigmoid'))

    model.build(input_shape)
    
    model.train(
        X_train, y_train,
        X_valid, y_valid,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    model.save(args.save_model)
    
if __name__ == "__main__":
    main()