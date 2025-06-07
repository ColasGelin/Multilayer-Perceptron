import numpy as np
import pandas as pd
import argparse
from typing import List, Dict
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from qbstyles import mpl_style

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1) 
        return self
    
    def transform(self, X):
        # Prevent division by zero
        scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return (X - self.mean_) / scale
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class DenseLayer:
    def __init__(self, units: int, activation: str = 'sigmoid'):
        self.units = units
        self.activation = activation
        self.weights = None
        self.bias = None
        self.m_weights = None
        self.m_bias = None
        self.v_weights = None
        self.v_bias = None
        self.input_shape = None

    def initialize(self, input_shape: int):
        self.input_shape = input_shape
        # He initialization
        stddev = np.sqrt(2.0 / self.input_shape)
        self.weights = np.random.randn(self.input_shape, self.units) * stddev
        self.bias = np.zeros((1, self.units))
        self.velocity_weights = np.zeros_like(self.weights) 
        self.velocity_bias = np.zeros_like(self.bias)
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        if self.activation == 'softmax':
            # subtracting max to prevent overflow
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'leaky_relu':
            self.alpha = 0.01 # or make it a parameter
            self.output = np.where(self.z > 0, self.z, self.z * self.alpha)
        else:  # sigmoid
            self.output = 1 / (1 + np.exp(-self.z))
            
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.activation == 'softmax':
            grad_z = grad_output
        elif self.activation == 'relu':
            grad_activation = np.where(self.z > 0, 1, 0)
            grad_z = grad_output * grad_activation
        elif self.activation == 'leaky_relu':
            grad_activation = np.where(self.z > 0, 1, self.alpha)
            grad_z = grad_output * grad_activation
        else:  # sigmoid
            grad_activation = self.output * (1 - self.output)
            grad_z = grad_output * grad_activation
            
        batch_size = self.inputs.shape[0]
        self.grad_weights = np.dot(self.inputs.T, grad_z) / batch_size
        self.grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        
        grad_input = np.dot(grad_z, self.weights.T)
        return grad_input
    
    def update_params_adam(self, learning_rate: float, beta1: float = 0.95, 
                            beta2: float = 0.999, epsilon: float = 1e-8, t: int = 1):
            # Update first moment (momentum)
            self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.grad_weights
            self.m_bias = beta1 * self.m_bias + (1 - beta1) * self.grad_bias
            
            # Update second moment (RMSprop)
            self.v_weights = beta2 * self.v_weights + (1 - beta2) * np.square(self.grad_weights)
            self.v_bias = beta2 * self.v_bias + (1 - beta2) * np.square(self.grad_bias)
            
            # Bias correction
            m_weights_corrected = self.m_weights / (1 - beta1**t)
            m_bias_corrected = self.m_bias / (1 - beta1**t)
            
            v_weights_corrected = self.v_weights / (1 - beta2**t)
            v_bias_corrected = self.v_bias / (1 - beta2**t)
            
            # Update weights and bias
            self.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
            self.bias -= learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + epsilon)

class MultiLayerPerceptron:
    def __init__(self, scaler=None):
        self.layers = []
        self.metrics_history = {
            'loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [], 
            'train_f1': [],
            'val_f1': []
        }
        self.adam_t = 0
        self.scaler = scaler if scaler is not None else CustomStandardScaler()

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

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, l2_lambda: float = 0.01) -> float:
        batch_size = y_true.shape[0]
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        base_loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        
        l2_reg_term = 0
        for layer in self.layers:
            l2_reg_term += np.sum(np.square(layer.weights))
        l2_loss = (l2_lambda / 2) * l2_reg_term / batch_size
        
        total_loss = base_loss + l2_loss
        
        grad_output = y_pred - y_true
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
            
            if hasattr(layer, 'weights') and layer.weights is not None:
                layer.grad_weights += l2_lambda * layer.weights / batch_size
        
        return total_loss
    
    def update_params(self, learning_rate: float):
        t = 1
        for layer in self.layers:
            layer.update_params_adam(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=t)
        t += 1

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, # y_val & y_train are one_hot encoded
              epochs: int = 1000, learning_rate: float = 0.001,
              batch_size: int = 16,
              early_stopping_patience: int = 50,
              min_delta: float = 0.0001,
              momentum_coeff: float = 0.9,
              plotting_enabled: bool = False,
              ) -> Dict[str, List[float]]:

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
                self.update_params(learning_rate)
                
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
        return probs[:, 1].reshape(-1, 1)  # Return positive class probability
    
    def save(self, filepath: str):
        model_data = []
        
        # Save scaler data first
        scaler_data = {
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_,
        }
        model_data.append(scaler_data)
        
        # Save layer data
        for layer in self.layers:
            layer_data = {
                'weights': layer.weights,
                'bias': layer.bias,
                'activation': layer.activation,
                'units': layer.units,
            }
            model_data.append(layer_data)
        
        np.save(filepath, model_data)
        
    def load(self, filepath: str):
        model_data = np.load(filepath, allow_pickle=True)
        
        # Load scaler data (first element)
        scaler_data = model_data[0]
        self.scaler.mean_ = scaler_data['mean']
        self.scaler.scale_ = scaler_data['scale']
        
        # Load layer data (remaining elements)
        layer_data_list = model_data[1:]
        
        if len(layer_data_list) != len(self.layers):
            raise ValueError(f"Model architecture mismatch: saved model has {len(layer_data_list)} layers, but current model has {len(self.layers)} layers")
        
        for i, layer_data in enumerate(layer_data_list):
            self.layers[i].weights = layer_data['weights']
            self.layers[i].bias = layer_data['bias']
            self.layers[i].activation = layer_data['activation']
            self.layers[i].units = layer_data['units']
            
            if i > 0: 
                self.layers[i].input_shape = self.layers[i-1].units
                
def load_scaler(filepath: str):
    model_data = np.load(filepath, allow_pickle=True)
    scaler_data = model_data[0]  # First element should be scaler data
    
    scaler = CustomStandardScaler()
    scaler.mean_ = scaler_data['mean']
    scaler.scale_ = scaler_data['scale']
    
    return scaler
            
def binary_cross_entropy(y_true, y_pred, class_weights=None):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    if class_weights is not None:
        sample_weights = np.zeros(y_true.shape[0])
        for i in range(len(class_weights)):
            sample_weights += (y_true == i) * class_weights[i]
        
        bce = -np.sum(sample_weights * (y_true * np.log(y_pred) + 
               (1 - y_true) * np.log(1 - y_pred))) / np.sum(sample_weights)
    else:
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
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    return X, y

def preprocess_data(data, scaler, fit=True):
    col_names = ['id']
    col_names.extend(['diagnosis'])
    col_names.extend([f'feature_{i}' for i in range(30)])
    data.columns = col_names[:len(data.columns)]

    data = data.drop('id', axis=1)

    # Convert diagnosis to binary
    diagnosis_mapping = {'M': 1, 'B': 0}
    data['diagnosis'] = data['diagnosis'].map(diagnosis_mapping)
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    if fit:
        X_train_scaled = scaler.fit_transform(X)
    else:
        X_train_scaled = scaler.transform(X)

    train_std_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_std_df['diagnosis'] = y.values

    print(f"Repartition - Benign: {(y == 0).sum()} ({(y == 0).sum()/len(y):.1%}), Malignant: {(y == 1).sum()} ({(y == 1).sum()/len(y):.1%})")

    return train_std_df

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
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'split'], 
                        help='Mode to run: train or predict')
    default_data = 'datasets/Validation.csv' if '--mode' in sys.argv and 'predict' in sys.argv else 'datasets/data.csv'
    parser.add_argument('--data', type=str, default=default_data, help='Path to data CSV file for prediction')
    parser.add_argument('--train', type=str, default='datasets/Training.csv', help='Path to training CSV file')
    parser.add_argument('--valid', type=str, default='datasets/Validation.csv', help='Path to validation CSV file')
    parser.add_argument('--layer', type=int, nargs='+', default=[24, 24], help='Number of units in each hidden layer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='output/model.npy', help='Path to save/load model')
    parser.add_argument('--esp', type=int, default=50, help='Early stopping patience')
    parser.add_argument('-p', action='store_const', const=True, default=False, help='Enable plotting of decision boundaries')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum coefficient')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_mode(args, parser)
    elif args.mode == 'predict':
        predict_mode(args, parser)
    else:
        split_mode(args.data)
        
def split_mode(data_path):
    data = pd.read_csv(data_path)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.75 * len(data))
    train_data = data.iloc[:split_idx]
    valid_data = data.iloc[split_idx:]
    train_data.to_csv('datasets/Training.csv', index=False)
    valid_data.to_csv('datasets/Validation.csv', index=False)
    
def train_mode(args, parser):
    if not args.train or not args.valid:
        parser.error("train mode requires --train and --valid")
    
    train_data = pd.read_csv(args.train)
    valid_data = pd.read_csv(args.valid)
    
    scaler = CustomStandardScaler()
    train_data_formated = preprocess_data(train_data, scaler)
    valid_data_formated = preprocess_data(valid_data, scaler, False)
    X_train, y_train = split_data(train_data_formated)
    X_valid, y_valid = split_data(valid_data_formated)

    # Convert to one-hot encoding for categorical cross-entropy
    y_train_one_hot = np.zeros((y_train.shape[0], 2))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train.flatten().astype(int)] = 1
    y_valid_one_hot = np.zeros((y_valid.shape[0], 2))
    y_valid_one_hot[np.arange(y_valid.shape[0]), y_valid.flatten().astype(int)] = 1
    
    print(f'x_train shape : {X_train.shape}')
    print(f'y_train shape : {y_train_one_hot.shape}')
    print(f'x_valid shape : {X_valid.shape}')
    print(f'y_valid shape : {y_valid_one_hot.shape}')
    
    model = MultiLayerPerceptron(scaler)
    
    input_shape = X_train.shape[1]
    for units in args.layer:
        model.add(DenseLayer(units, activation='relu'))
    model.add(DenseLayer(2, activation='softmax'))

    model.build(input_shape)
    
    history = model.train(
        X_train, y_train_one_hot,
        X_valid, y_valid_one_hot,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        plotting_enabled=args.p,
        early_stopping_patience=args.esp,
        momentum_coeff=args.momentum,
    )
    
    model.save(args.model)
    
    plot_learning_curves(history)

def predict_mode(args, parser):
    if not args.data or not args.model:
        parser.error("predict mode requires --data and --model")
    
    data = pd.read_csv(args.data)
    data_formated = preprocess_data(data, load_scaler(args.model), False)
    X, y_1d = split_data(data_formated)

    y = y_1d.reshape(-1, 1)
    
    # Pass the loaded scaler to the model
    model = MultiLayerPerceptron(load_scaler(args.model))

    input_shape = X.shape[1]
    for units in args.layer:
        model.add(DenseLayer(units, activation='relu'))
    model.add(DenseLayer(2, activation='softmax'))
    
    model.build(input_shape)
    model.load(args.model)
    
    positive_probs = model.predict(X)
    
    bce = binary_cross_entropy(y, positive_probs)
    print(f"Binary Cross-Entropy: {bce:.4f}")
    
    predicted_classes = (positive_probs >= 0.5).astype(int)
    accuracy = np.mean(predicted_classes == y)
    print(f"Accuracy: {accuracy:.4f}")
    f1_score = calculate_f1_score(y, predicted_classes, True)
    print(f"F1 Score: {f1_score:.4f}")

def plot_decision_boundary_epoch(model: 'MultiLayerPerceptron',
                                 X_data_full: np.ndarray,
                                 y_data_one_hot: np.ndarray,
                                 epoch: int):
    
    feature_pairs_to_plot = [(0, i) for i in range(1, 16)]
    
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15)) 
    fig.suptitle(f'Decision Boundaries - Epoch {epoch + 1}', fontsize=16)

    mean_fill_values_for_plot = np.mean(X_data_full, axis=0)

    for i, ax in enumerate(axes.flat):

        idx1, idx2 = feature_pairs_to_plot[i]

        X_plot_subset = X_data_full[:, [idx1, idx2]]
        y_labels = np.argmax(y_data_one_hot, axis=1) 

        x_min, x_max = X_plot_subset[:, 0].min() - 0.5, X_plot_subset[:, 0].max() + 0.5
        y_min, y_max = X_plot_subset[:, 1].min() - 0.5, X_plot_subset[:, 1].max() + 0.5
        h = 0.05 
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        mesh_input = np.tile(mean_fill_values_for_plot, (xx.ravel().shape[0], 1))
        mesh_input[:, idx1], mesh_input[:, idx2] = xx.ravel(), yy.ravel()

        Z_probs = model.forward(mesh_input)
        Z = Z_probs[:, 1] if Z_probs.shape[1] == 2 else (Z_probs.ravel() if Z_probs.shape[1] == 1 else Z_probs[:,1]) 
        Z = Z.reshape(xx.shape)
        
        contour_levels = [0.1, 0.25, 0.5, 0.75, 0.9] 
        cs = ax.contourf(xx, yy, Z, levels=contour_levels, cmap="coolwarm", linewidths=1, alpha=0.5) 
        ax.clabel(cs, inline=True, fontsize=8, fmt='P=%.2f')

        ax.scatter(X_plot_subset[:, 0], X_plot_subset[:, 1], c=y_labels,
                              cmap=plt.cm.coolwarm, s=20, edgecolor='k', alpha=0.9) 

        subplot_title_str = f'Features {idx1+1} & {idx2+1}' 
        x_label_str = f'Feature {idx1+1}' 
        y_label_str = f'Feature {idx2+1}' 
        
        ax.set_title(subplot_title_str, fontsize=10)
        ax.set_xlabel(x_label_str, fontsize=9); ax.set_ylabel(y_label_str, fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout() 
    num_digits = 3
    filename = f"output/epochs/decision_boundary_grid_epoch_{epoch + 1:0{num_digits}d}.png"
    plt.savefig(filename)
    plt.close(fig) 
    print(f"Saved decision boundary grid plot to {filename}")

if __name__ == "__main__":
    main()
    
