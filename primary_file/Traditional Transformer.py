import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('merged_data.csv')

# Define target columns
target_columns = [
    'pose.position.x', 'pose.position.y', 'pose.position.z',
    'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w'
]

# Extract input features (excluding 'Time' and target columns)
input_columns = data.columns.difference(['Time'] + target_columns)
inputs = data[input_columns].values

# Extract target columns
targets = data[target_columns].values

# Check for NaNs in the data
if data.isnull().values.any():
    raise ValueError("Data contains NaNs. Please handle them before training.")
else:
    print("No NaNs found in the data.")

# Normalize the input data
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and add sequence dimension
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class PoseTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(PoseTransformer, self).__init__()
        self.model_dim = model_dim

        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, model_dim))  # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True)
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


# Hyperparameters
input_dim = X_train_tensor.shape[2]
model_dim = 64
num_heads = 8
num_layers = 4
output_dim = len(target_columns)
dropout = 0.1

# Initialize model, loss function, and optimizer
model = PoseTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move model to GPU
model.to(device)


# Function to compute metrics
def compute_metrics(outputs, targets):
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    rmse = mean_squared_error(targets_np, outputs_np, squared=False)
    mae = mean_absolute_error(targets_np, outputs_np)
    r2 = r2_score(targets_np, outputs_np)

    return rmse, mae, r2


# Training function
def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=150):
    train_rmse_list, test_rmse_list = [], []
    train_mae_list, test_mae_list = [], []
    train_r2_list, test_r2_list = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        if (epoch + 1) % 2 == 0:
            # Compute train metrics
            model.eval()
            train_outputs, train_targets = [], []
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    train_outputs.append(outputs)
                    train_targets.append(targets)
            train_outputs = torch.cat(train_outputs)
            train_targets = torch.cat(train_targets)
            train_rmse, train_mae, train_r2 = compute_metrics(train_outputs, train_targets)
            train_rmse_list.append(train_rmse)
            train_mae_list.append(train_mae)
            train_r2_list.append(train_r2)

            # Compute test metrics
            test_outputs, test_targets = [], []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_outputs.append(outputs)
                    test_targets.append(targets)
            test_outputs = torch.cat(test_outputs)
            test_targets = torch.cat(test_targets)
            test_rmse, test_mae, test_r2 = compute_metrics(test_outputs, test_targets)
            test_rmse_list.append(test_rmse)
            test_mae_list.append(test_mae)
            test_r2_list.append(test_r2)

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}, Train R2: {train_r2:.4f}, Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')

    # Plotting metrics
    epochs = list(range(2, num_epochs + 1, 2))

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_rmse_list, label='Train RMSE')
    plt.plot(epochs, test_rmse_list, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE Over Epochs')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mae_list, label='Train MAE')
    plt.plot(epochs, test_mae_list, label='Test MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE Over Epochs')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_r2_list, label='Train R2')
    plt.plot(epochs, test_r2_list, label='Test R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title('R2 Over Epochs')

    plt.tight_layout()
    plt.show()


# Train the model
train_model(model, criterion, optimizer, train_loader, test_loader)

# Save the model
torch.save(model.state_dict(), 'pose_transformer.pth')


# Function to plot ground truth vs. predictions
def plot_ground_truth_vs_predictions(model, test_loader, target_columns):
    model.eval()
    test_outputs, test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_outputs.append(outputs)
            test_targets.append(targets)
    test_outputs = torch.cat(test_outputs).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()

    for i, target_column in enumerate(target_columns):
        plt.figure()
        plt.plot(test_targets[:, i], label='Ground Truth')
        plt.plot(test_outputs[:, i], label='Prediction')
        plt.xlabel('Sample')
        plt.ylabel(target_column)
        plt.title(f'Ground Truth vs. Prediction for {target_column}')
        plt.legend()
        plt.show()


# Plot ground truth vs. predictions
plot_ground_truth_vs_predictions(model, test_loader, target_columns)
