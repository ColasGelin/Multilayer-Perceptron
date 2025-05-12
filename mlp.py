import numpy as np
import pandas as pd
import argparse
from typing import List, Dict
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from qbstyles import mpl_style


class DenseLayer:
    def __init__(self, units: int, activation: str = 'sigmoid'):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.input_shape = None
        self.velocity_weights = None 
        self.velocity_bias = None 
        
    def initialize(self, input_shape: int):
        self.input_shape = input_shape
        limit = np.sqrt(6 / self.input_shape)
        self.weights = np.random.uniform(-limit, limit, (self.input_shape, self.units))
        self.bias = np.zeros((1, self.units))
        self.velocity_weights = np.zeros_like(self.weights) 
        self.velocity_bias = np.zeros_like(self.bias)
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'softmax':
            # subtracting max to prevent overflow
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:  # sigmoid
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
    
    def update_params(self, learning_rate: float, momentum_coeff: float = 0.0):
        # Momentum update
        self.velocity_weights = (momentum_coeff * self.velocity_weights) - (learning_rate * self.grad_weights)
        self.velocity_bias = (momentum_coeff * self.velocity_bias) - (learning_rate * self.grad_bias)
        
        self.weights += self.velocity_weights
        self.bias += self.velocity_bias

class MultiLayerPerceptron:
    def __init__(self):
        self.layers = []
        self.metrics_history = {
            'loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [], 
            'train_f1': [],
            'val_f1': []
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
        
        # Preventing log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        grad_output = y_pred - y_true
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
            
        return loss
    
    def update_params(self, learning_rate: float, momentum_coeff: float):
            for layer in self.layers:
                layer.update_params(learning_rate, momentum_coeff)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, # y_val & y_train are one_hot encoded
              epochs: int = 10, learning_rate: float = 0.001,
              batch_size: int = 16,
              early_stopping_patience: int = 10,
              min_delta: float = 0.0001,
              momentum_coeff: float = 0.,
              plotting_enabled: bool = False) -> Dict[str, List[float]]:

        num_samples = X_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                y_pred = self.forward(X_batch)
                batch_loss = self.backward(y_batch, y_pred)
                self.update_params(learning_rate, momentum_coeff)
                
                epoch_loss += batch_loss
            epoch_loss /= num_batches

            # Preventing log(0) 
            epsilon = 1e-15
            y_val_pred = self.forward(X_val)
            y_val_pred = np.clip(y_val_pred, epsilon, 1 - epsilon)
            y_train_pred = self.forward(X_train)
            y_train_pred = np.clip(y_train_pred, epsilon, 1 - epsilon)

            # Converting one-hot encoded labels to class indices
            true_classes_val = np.argmax(y_val, axis=1) 
            predicted_classes_val = np.argmax(y_val_pred, axis=1)
            true_classes_train = np.argmax(y_train, axis=1) 
            predicted_classes_train = np.argmax(y_train_pred, axis=1)

            val_loss = -np.sum(y_val * np.log(y_val_pred)) / y_val.shape[0]
            self.metrics_history['loss'].append(epoch_loss)
            self.metrics_history['val_loss'].append(val_loss)

            train_accuracy = np.mean(true_classes_train == predicted_classes_train)
            val_accuracy = np.mean(true_classes_val == predicted_classes_val)
            self.metrics_history['train_accuracy'].append(train_accuracy)
            self.metrics_history['val_accuracy'].append(val_accuracy)

            val_f1 = calculate_f1_score(true_classes_val, predicted_classes_val)
            train_f1 = calculate_f1_score(true_classes_train, predicted_classes_train)
            self.metrics_history['train_f1'].append(train_f1)
            self.metrics_history['val_f1'].append(val_f1)

            print(f'epoch {epoch+1:02d}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - val_f1: {val_f1:.4f}')

            # Plotting decision boundaries every 10 epochs
            if (plotting_enabled):
                if (epoch + 1) % 10 == 0:  
                    plot_decision_boundary_epoch(
                        model=self,
                        X_data_full=X_val,         
                        y_data_one_hot=y_val,
                        epoch=epoch
                    )

            # Early stopping
            if early_stopping_patience is not None:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1} as validation loss did not improve for {early_stopping_patience} epochs.")
                        break
                    
        return self.metrics_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
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
        
    def load(self, filepath: str):
        model_data = np.load(filepath, allow_pickle=True)
        
        if len(model_data) != len(self.layers):
            raise ValueError(f"Model architecture mismatch: saved model has {len(model_data)} layers, but current model has {len(self.layers)} layers")
        
        for i, layer_data in enumerate(model_data):
            self.layers[i].weights = layer_data['weights']
            self.layers[i].bias = layer_data['bias']
            self.layers[i].activation = layer_data['activation']
            self.layers[i].units = layer_data['units']
            
            if i > 0: 
                self.layers[i].input_shape = self.layers[i-1].units
            
        print(f"> loaded model from '{filepath}'")

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate binary cross-entropy error as specified in the project"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    N = y_true.shape[0]
    bce = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / N
    
    return bce

def plot_learning_curves(history: Dict[str, List[float]]):
    epochs_range = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output/plots.png")

def split_data(data):
    # Extract features data ([:, :-1] selects all rows and all column except the last one)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    return X, y

def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, plot: bool = False) -> float:
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."
    assert np.all(np.isin(y_true, [0, 1])), "True labels must be binary (0 or 1)."
    assert np.all(np.isin(y_pred, [0, 1])), "Predicted labels must be binary (0 or 1)."

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    if (plot):
        confusion_matrix = np.array([[tn, fp], 
                                    [fn, tp]])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted Maligne', 'Predicted Maligne'], 
                    yticklabels=['Actual Benign', 'Actual benign'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("Confusion Matrix")
        plt.savefig('output/confusion_matrix.png')
    return f1

def main():
    mpl_style(dark=True)
    
    parser = argparse.ArgumentParser(description='Multilayer perceptron for breast cancer classification')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help='Mode to run: train or predict')
    default_data = 'datasets/Validation.csv' if '--mode' in sys.argv and 'predict' in sys.argv else 'datasets/data.csv'
    parser.add_argument('--data', type=str, default=default_data, help='Path to data CSV file for prediction')
    parser.add_argument('--train', type=str, default='datasets/Training.csv', help='Path to training CSV file')
    parser.add_argument('--valid', type=str, default='datasets/Validation.csv', help='Path to validation CSV file')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help='Number of units in each hidden layer')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='output/model.npy', help='Path to save/load model')
    parser.add_argument('-p', action='store_const', const=True, default=False, help='Enable plotting of decision boundaries')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args, parser)
    elif args.mode == 'predict':
        predict_mode(args, parser)
    
def train_mode(args, parser):
    if not args.train or not args.valid:
        parser.error("train mode requires --train and --valid")
    
    train_data = pd.read_csv(args.train)
    valid_data = pd.read_csv(args.valid)
    
    X_train, y_train = split_data(train_data)
    X_valid, y_valid = split_data(valid_data)
    
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
    
    history = model.train(
        X_train, y_train_one_hot,
        X_valid, y_valid_one_hot,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        plotting_enabled=args.p,
    )
    
    model.save(args.model)
    
    plot_learning_curves(history)

def predict_mode(args, parser):
    if not args.data or not args.model:
        parser.error("predict mode requires --data and --model")
    
    data = pd.read_csv(args.data)
    X, y = split_data(data)
    
    model = MultiLayerPerceptron()
    
    input_shape = X.shape[1]
    for units in args.layer:
        model.add(DenseLayer(units, activation='sigmoid'))
    model.add(DenseLayer(2, activation='softmax'))
    
    model.build(input_shape)
    model.load(args.model)
    
    
    positive_probs = model.predict(X)
    
    bce = binary_cross_entropy(y, positive_probs)
    print(f"Binary Cross-Entropy: {bce:.4f}")
    
    predicted_classes = (positive_probs >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)
    calculate_f1_score(y, predicted_classes, True)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nSample predictions (First 5):")
    print("True\tPred\tProbability")
    for i in range(min(5, len(y))):
        print(f"{y[i, 0]}\t{predicted_classes[i, 0]}\t{positive_probs[i, 0]:.4f}")

def plot_decision_boundary_epoch(model: 'MultiLayerPerceptron',
                                 X_data_full: np.ndarray,
                                 y_data_one_hot: np.ndarray,
                                 epoch: int):
    
    num_total_features = X_data_full.shape[1]
    feature_pairs_to_plot = [(0, i) for i in range(1, 16)]
    
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15)) 
    fig.suptitle(f'Decision Boundaries - Epoch {epoch + 1}', fontsize=16)

    mean_fill_values_for_plot = None
    if num_total_features > 2:
        mean_fill_values_for_plot = np.mean(X_data_full, axis=0)

    for i, ax in enumerate(axes.flat):
        if i >= len(feature_pairs_to_plot):
            ax.axis('off') 
            continue

        idx1, idx2 = feature_pairs_to_plot[i]
        
        if idx2 >= num_total_features:
            ax.set_title(f"Pair [{idx1+1},{idx2+1}] N/A") # Use 1-based indexing for display
            ax.axis('off')
            continue

        X_plot_subset = X_data_full[:, [idx1, idx2]]
        y_labels = np.argmax(y_data_one_hot, axis=1) 

        x_min, x_max = X_plot_subset[:, 0].min() - 0.5, X_plot_subset[:, 0].max() + 0.5
        y_min, y_max = X_plot_subset[:, 1].min() - 0.5, X_plot_subset[:, 1].max() + 0.5
        h = 0.05 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        if num_total_features > 2:
            mesh_input = np.tile(mean_fill_values_for_plot, (xx.ravel().shape[0], 1))
            mesh_input[:, idx1], mesh_input[:, idx2] = xx.ravel(), yy.ravel()
        else: 
            mesh_input = np.c_[xx.ravel(), yy.ravel()]

        Z_probs = model.forward(mesh_input)
        Z = Z_probs[:, 1] if Z_probs.shape[1] == 2 else (Z_probs.ravel() if Z_probs.shape[1] == 1 else Z_probs[:,1]) 
        Z = Z.reshape(xx.shape)
        
        
        contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9] 
        cs = ax.contourf(xx, yy, Z, levels=contour_levels, cmap="coolwarm", linewidths=1, alpha=0.5) 
        ax.clabel(cs, inline=True, fontsize=8, fmt='P=%.2f')

        ax.scatter(X_plot_subset[:, 0], X_plot_subset[:, 1], c=y_labels,
                              cmap=plt.cm.coolwarm, s=20, edgecolor='k', alpha=0.9) 

        subplot_title_str = f'Features {idx1+1} & {idx2+1}' # Use 1-based indexing for display
        x_label_str = f'Feature {idx1+1}' # Use 1-based indexing for display
        y_label_str = f'Feature {idx2+1}' # Use 1-based indexing for display
        
        ax.set_title(subplot_title_str, fontsize=10) # Use the ensured numerical title
        ax.set_xlabel(x_label_str, fontsize=9); ax.set_ylabel(y_label_str, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    num_digits = 3
    filename = f"output/epochs/decision_boundary_grid_epoch_{epoch + 1:0{num_digits}d}.png"
    plt.savefig(filename)
    plt.close(fig) 
    print(f"Saved decision boundary grid plot to {filename}")

if __name__ == "__main__":
    main()
    
