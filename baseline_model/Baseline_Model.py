import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load IMU data and ground truth data from CSV files
imu_data = pd.read_csv('/Users/dingshengliu/Desktop/APS360/downsampled_imu_data.csv')
ground_truth = pd.read_csv('/Users/dingshengliu/Desktop/APS360/new_truth.csv')

# Time step (s)
dt = 1 / 100

# Get initial position from the first row of the ground truth data
initial_position = ground_truth.iloc[0][['pose.position.x', 'pose.position.y', 'pose.position.z']].to_numpy()

# Initialize position, velocity, orientation (quaternions), and angular velocity
position = initial_position.copy()
velocity = np.array([0.0, 0.0, 0.0])
orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z)
angular_velocity = np.array([0.0, 0.0, 0.0])

# Initialize lists to store calculated and true positions
calculated_positions = []

# Initial alpha value
alpha = 0.99

# Helper function to update orientation using quaternion multiplication
def quaternion_multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# Helper function to normalize a quaternion
def normalize_quaternion(q):
    return q / np.linalg.norm(q)

# Loop through IMU data
for index, row in imu_data.iterrows():
    ax, ay, az = row['accel.linear.x'], row['accel.linear.y'], row['accel.linear.z']
    wx, wy, wz = row['accel.angular.x'], row['accel.angular.y'], row['accel.angular.z']

    # Update angular velocity
    angular_velocity[0] = wx * dt
    angular_velocity[1] = wy * dt
    angular_velocity[2] = wz * dt

    # Convert angular velocity to quaternion
    theta = np.linalg.norm(angular_velocity)
    if theta > 0:
        axis = angular_velocity / theta
        sin_half_theta = np.sin(theta / 2.0)
        delta_q = np.array([
            np.cos(theta / 2.0),
            axis[0] * sin_half_theta,
            axis[1] * sin_half_theta,
            axis[2] * sin_half_theta
        ])
    else:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0])

    # Update orientation using quaternion multiplication
    orientation = quaternion_multiply(orientation, delta_q)
    orientation = normalize_quaternion(orientation)

    # Rotate linear acceleration to the global frame using the orientation quaternion
    rot_matrix = np.array([
        [1 - 2*(orientation[2]**2 + orientation[3]**2), 2*(orientation[1]*orientation[2] - orientation[3]*orientation[0]), 2*(orientation[1]*orientation[3] + orientation[2]*orientation[0])],
        [2*(orientation[1]*orientation[2] + orientation[3]*orientation[0]), 1 - 2*(orientation[1]**2 + orientation[3]**2), 2*(orientation[2]*orientation[3] - orientation[1]*orientation[0])],
        [2*(orientation[1]*orientation[3] - orientation[2]*orientation[0]), 2*(orientation[2]*orientation[3] + orientation[1]*orientation[0]), 1 - 2*(orientation[1]**2 + orientation[2]**2)]
    ])
    accel_global = rot_matrix @ np.array([ax, ay, az])

    # Update velocity
    velocity[0] += accel_global[0] * dt
    velocity[1] += accel_global[1] * dt
    velocity[2] += accel_global[2] * dt

    # Calculate new position
    new_position = position + velocity * dt

    # Apply complementary filter to reduce drift with decreasing alpha
    position = alpha * new_position + (1 - alpha) * initial_position

    # Append calculated position
    calculated_positions.append(position.copy())

    # Calculate the magnitude of the acceleration vector
    accel_magnitude = np.linalg.norm([ax, ay, az])

# Convert calculated_positions to a DataFrame
calculated_positions_df = pd.DataFrame(calculated_positions, columns=['x', 'y', 'z'])

# Extract ground truth positions and flip Y-coordinate direction
ground_truth_positions = ground_truth[['pose.position.x', 'pose.position.y', 'pose.position.z']].copy()
ground_truth_positions['pose.position.y'] = -ground_truth_positions['pose.position.y']
ground_truth_positions.columns = ['x_true', 'y_true', 'z_true']

# Combine calculated positions with ground truth
results = pd.concat([ground_truth[['Time']], ground_truth_positions, calculated_positions_df], axis=1)
results.columns = ['Time', 'x_true', 'y_true', 'z_true', 'x_pred', 'y_pred', 'z_pred']

# Drop any rows with NaN values
results = results.dropna()

# Calculate mean squared error for each coordinate
mse_x = mean_squared_error(results['x_true'], results['x_pred'])
mse_y = mean_squared_error(results['y_true'], results['y_pred'])
mse_z = mean_squared_error(results['z_true'], results['z_pred'])

# Print the results
print(f"Mean Squared Error (X): {mse_x}")
print(f"Mean Squared Error (Y): {mse_y}")
print(f"Mean Squared Error (Z): {mse_z}")

# Save results to a CSV file
results.to_csv('comparison_results.csv', index=False)

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot x_true vs x_pred
plt.subplot(3, 1, 1)
plt.plot(results['Time'], results['x_true'], label='True X')
plt.plot(results['Time'], results['x_pred'], label='Predicted X')
plt.xlabel('Time')
plt.ylabel('X Coordinate')
plt.legend()
plt.title('True vs Predicted X Coordinate')

# Plot y_true vs y_pred
plt.subplot(3, 1, 2)
plt.plot(results['Time'], results['y_true'], label='True Y')
plt.plot(results['Time'], results['y_pred'], label='Predicted Y')
plt.xlabel('Time')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title('True vs Predicted Y Coordinate')

# Plot z_true vs z_pred
plt.subplot(3, 1, 3)
plt.plot(results['Time'], results['z_true'], label='True Z')
plt.plot(results['Time'], results['z_pred'], label='Predicted Z')
plt.xlabel('Time')
plt.ylabel('Z Coordinate')
plt.legend()
plt.title('True vs Predicted Z Coordinate')

plt.tight_layout()
plt.show()
