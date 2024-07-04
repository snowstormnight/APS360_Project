import pandas as pd
import numpy as np

# Load IMU and position data from CSV files
imu_df = pd.read_csv("D:\Study\Summer 2024\APS360\Project\Primary Model\Circle.csv")  # Replace with the actual file name
position_df = pd.read_csv("D:\Study\Summer 2024\APS360\Project\Primary Model\Circle_truth.csv")  # Replace with the actual file name

# Load the IMU data (replace 'imu_data.csv' with the actual file name)
imu_data = pd.read_csv("D:\Study\Summer 2024\APS360\Project\Primary Model\Circle.csv")

# Load the ground truth position data (replace 'groundtruth_data.csv' with the actual file name)
groundtruth_data = pd.read_csv("D:\Study\Summer 2024\APS360\Project\Primary Model\Circle_truth.csv")

# Calculate the downsampling factor
downsample_factor = 380 // 96

# Downsample the IMU data by selecting every 9th row
downsampled_imu_data = imu_data.iloc[::downsample_factor]

# Ensure the downsampled data has the same number of rows as the ground truth data
if len(downsampled_imu_data) > len(groundtruth_data):
    downsampled_imu_data = downsampled_imu_data.iloc[:len(groundtruth_data)]
elif len(downsampled_imu_data) < len(groundtruth_data):
    groundtruth_data = groundtruth_data.iloc[:len(downsampled_imu_data)]

# Reset the index of both dataframes to ensure alignment
downsampled_imu_data.reset_index(drop=True, inplace=True)
groundtruth_data.reset_index(drop=True, inplace=True)

# Combine the downsampled IMU data with the ground truth data
combined_data = pd.concat([downsampled_imu_data, groundtruth_data], axis=1)

# Save the combined data to a new CSV file
combined_data.to_csv('combined_imu_groundtruth.csv', index=False)

print("Downsampling and combining completed. Saved to 'combined_imu_groundtruth.csv'.")

