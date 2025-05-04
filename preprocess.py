import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Calculate mean and standard deviation for each feature."""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=1) 
        return self
    
    def transform(self, X):
        """Standardize features by removing the mean and scaling to unit variance."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Prevent division by zero
        scale = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return (X - self.mean_) / scale
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Main processing script
if len(sys.argv) != 2:
    print("Usage: split.py <input_csv>")
    sys.exit(1)

input_csv = sys.argv[1]

if not os.path.isfile(input_csv):
    print(f"File '{input_csv}' does not exist.")
    sys.exit(1)

# Load data
data = pd.read_csv(input_csv, header=None)

# Give appropriate column names
col_names = ['id']
col_names.extend(['diagnosis'])
col_names.extend([f'feature_{i}' for i in range(30)])
data.columns = col_names[:len(data.columns)]
print(f"Dataset shape: {data.shape}")

# Drop the ID column
data = data.drop('id', axis=1)

# Convert diagnosis to binary
diagnosis_mapping = {'M': 1, 'B': 0}
data['diagnosis'] = data['diagnosis'].map(diagnosis_mapping)
print(f"Class distribution - Benign: {(data['diagnosis'] == 0).sum()} ({(data['diagnosis'] == 0).sum()/len(data):.1%}), Malignant: {(data['diagnosis'] == 1).sum()} ({(data['diagnosis'] == 1).sum()/len(data):.1%})")

# Shuffle data with a fixed random seed
np.random.seed(42)
data = data.sample(frac=1).reset_index(drop=True)

# Split into training and validation sets (80-20 split)
split_idx = int(len(data) * 0.8)
training_data = data[:split_idx]
validation_data = data[split_idx:]

# Separate features and target for preprocessing
X_train = training_data.drop('diagnosis', axis=1)
y_train = training_data['diagnosis']
X_val = validation_data.drop('diagnosis', axis=1)
y_val = validation_data['diagnosis']

# Standardize features using only training data statistics
scaler = CustomStandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val) 

train_std_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_std_df['diagnosis'] = y_train.values
val_std_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
val_std_df['diagnosis'] = y_val.values
train_std_df.to_csv('datasets/Training.csv', index=False)
val_std_df.to_csv('datasets/Validation.csv', index=False)

# Print class distribution information
print(f"Training Repartition - Benign: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train):.1%}), Malignant: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train):.1%})")
print(f"Validation Repartition - Benign: {(y_val == 0).sum()} ({(y_val == 0).sum()/len(y_val):.1%}), Malignant: {(y_val == 1).sum()} ({(y_val == 1).sum()/len(y_val):.1%})")