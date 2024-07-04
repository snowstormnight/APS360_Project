import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Load the CSV file
file_path = "D:\Study\Summer 2024\APS360\Project\pythonProject\combined_imu_groundtruth.csv"  # Update with the correct path
data = pd.read_csv(file_path)

# Define input and target columns
input_columns = ['header.stamp.nsecs',
                 'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
                 'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z']
target_columns = ['pose.position.x', 'pose.position.y', 'pose.position.z',
                  'pose.orientation.x', 'pose.orientation.y', 'pose.orientation.z', 'pose.orientation.w']

# Normalize the data
scaler = StandardScaler()
data[input_columns + target_columns] = scaler.fit_transform(data[input_columns + target_columns])

# Convert to numpy arrays
X = data[input_columns].values
y = data[target_columns].values

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.65 * len(dataset))
eval_size = len(dataset) - train_size
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

# Create DataLoaders
batch_size = 64  # You can adjust this value
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM model
class DronePositionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_prob=0.1):
        super(DronePositionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_dim = len(input_columns)
hidden_dim = 50
output_dim = len(target_columns)
num_layers = 2
dropout_prob = 0.1

# Instantiate the model
model = DronePositionLSTM(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 200  # You can adjust this value
train_losses = []
eval_losses = []
train_maes = []
eval_maes = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_mae = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1)  # Add sequence dimension

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_mae += mean_absolute_error(targets.cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)

    epoch_train_loss /= len(train_loader.dataset)
    epoch_train_mae /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    train_maes.append(epoch_train_mae)

    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            epoch_eval_loss = 0.0
            epoch_eval_mae = 0.0
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                epoch_eval_loss += loss.item() * inputs.size(0)
                epoch_eval_mae += mean_absolute_error(targets.cpu().numpy(), outputs.detach().cpu().numpy()) * inputs.size(0)

            epoch_eval_loss /= len(eval_loader.dataset)
            epoch_eval_mae /= len(eval_loader.dataset)
            eval_losses.append(epoch_eval_loss)
            eval_maes.append(epoch_eval_mae)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Eval Loss: {epoch_eval_loss:.4f}, Train MAE: {epoch_train_mae:.4f}, Eval MAE: {epoch_eval_mae:.4f}")

# Evaluation and Plotting
model.eval()
with torch.no_grad():
    eval_predictions = []
    eval_targets = []
    for inputs, targets in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        eval_predictions.append(outputs.cpu().numpy())
        eval_targets.append(targets.cpu().numpy())

    eval_predictions = np.concatenate(eval_predictions, axis=0)
    eval_targets = np.concatenate(eval_targets, axis=0)

# # Plot the results
# time_steps = np.arange(len(eval_predictions))
# for i, col in enumerate(target_columns):
#     plt.figure(figsize=(14, 6))
#     plt.plot(time_steps, eval_targets[:, i], label='Actual', color='blue')
#     plt.plot(time_steps, eval_predictions[:, i], label='Predicted', color='red')
#     plt.title(f'Actual vs Predicted {col}')
#     plt.xlabel('Time Step')
#     plt.ylabel(col)
#     plt.legend()
#     plt.show()

# Plotting the losses
plt.figure(figsize=(14, 6))
plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(np.arange(1, num_epochs + 1, 2), eval_losses, label='Eval Loss', color='red', marker='o')
plt.title('Training and Evaluation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the MAE
plt.figure(figsize=(14, 6))
plt.plot(np.arange(1, num_epochs + 1), train_maes, label='Train MAE', color='blue')
plt.plot(np.arange(1, num_epochs + 1, 2), eval_maes, label='Eval MAE', color='red', marker='o')
plt.title('Training and Evaluation Mean Absolute Error (MAE)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
