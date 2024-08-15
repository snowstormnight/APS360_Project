import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define the PoseTransformer model
class PoseTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(PoseTransformer, self).__init__()
        self.model_dim = model_dim

        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, model_dim))  # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, output_dim)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.input_fc.weight, a=math.sqrt(5))
        if self.input_fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.input_fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.input_fc.bias, -bound, bound)
        nn.init.kaiming_uniform_(self.output_fc.weight, a=math.sqrt(5))
        if self.output_fc.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.output_fc.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.output_fc.bias, -bound, bound)

    def forward(self, x):
        x = self.input_fc(x)
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]  # Add positional encoding
        x = self.transformer_encoder(x)
        x = self.output_fc(x[:, -1, :])  # Only take the output of the last token
        return x

# Load the new dataset
new_data = pd.read_csv("evaluation_data2.csv")

# Define target columns (adjust based on new dataset)
target_columns = [
    'pose.position.x', 'pose.position.y', 'pose.position.z',
    'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'
]

# Extract input features (excluding 'Time' and target columns)
input_columns = new_data.columns.difference(['Time'] + target_columns)
inputs = new_data[input_columns].values
targets = new_data[target_columns].values

# Check for NaNs in the new data
if new_data.isnull().values.any():
    raise ValueError("Data contains NaNs. Please handle them before training.")
else:
    print("No NaNs found in the new data.")

# Normalize the input data using the same scaler
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and add sequence dimension
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader for training and validation
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the pre-trained model
input_dim = X_train_tensor.shape[2]
model_dim = 64
num_heads = 8
num_layers = 4
output_dim = len(target_columns)
dropout = 0.1

model = PoseTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)
model.load_state_dict(torch.load('pose_transformer.pth'))

# Move the model to GPU if available
device = torch.device('cpu')
model.to(device)

def plot_actual_vs_predictions_and_save_csv(model, data, input_columns, target_columns, scaler, csv_file_path):
    model.eval()

    # Extract inputs and actual targets
    inputs = data[input_columns].values
    actual_targets = data[target_columns].values

    # Normalize inputs using the same scaler
    inputs = scaler.transform(inputs)

    # Convert inputs to PyTorch tensor and add sequence dimension
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)

    # Generate predictions
    with torch.no_grad():
        inputs_tensor = inputs_tensor.to(device)  # Move inputs to the same device as the model
        predictions = model(inputs_tensor).cpu().numpy()

    # Compute metrics for each target column
    rmse = np.sqrt(mean_squared_error(actual_targets, predictions, multioutput='raw_values'))
    mae = mean_absolute_error(actual_targets, predictions, multioutput='raw_values')
    r2 = r2_score(actual_targets, predictions, multioutput='raw_values')

    # Print the metrics
    for i, target_column in enumerate(target_columns):
        print(f'Metrics for {target_column}:')
        print(f'  RMSE: {rmse[i]}')
        print(f'  MAE: {mae[i]}')
        print(f'  R²: {r2[i]}')
        print()

    # Create DataFrame with actual and predicted values
    results_df = pd.DataFrame(
        np.hstack([actual_targets, predictions]),
        columns=[f'Actual_{col}' for col in target_columns] + [f'Predicted_{col}' for col in target_columns]
    )

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_file_path, index=False)

    # Plot each target variable
    for i, target_column in enumerate(target_columns):
        plt.figure(figsize=(14, 6))
        plt.plot(actual_targets[:, i], label='Actual', alpha=0.6)
        plt.plot(predictions[:, i], label='Prediction', alpha=0.6)
        plt.xlabel('Sample')
        plt.ylabel(target_column)
        plt.title(f'Actual vs. Prediction for {target_column}')
        plt.legend()
        plt.show()
# Path to save the CSV file
csv_file_path = 'predictions_vs_actuals_new.csv'

# Plot actual vs predictions and save results to CSV
plot_actual_vs_predictions_and_save_csv(model, new_data, input_columns, target_columns, scaler, csv_file_path)

