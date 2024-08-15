import pandas as pd
import numpy as np

# Step 1: Read the CSV files
imu_data = pd.read_csv(r"D:\Study\Summer 2024\APS360\Project\pythonProject\leaf_fail_3\adr-bf-imu.csv")
position_data = pd.read_csv(r"D:\Study\Summer 2024\APS360\Project\pythonProject\leaf_fail_3\adr-vicon-adr_drone_1-pose.csv")

# Number of data rows (excluding header)
num_samples_160hz = len(imu_data)
num_samples_48hz = len(position_data)

# Calculate the upsampling factor
upsample_factor = num_samples_160hz / num_samples_48hz

# Initialize list to hold the new upsampled position data
upsampled_position_data = []

# Perform interpolation
for i in range(num_samples_48hz - 1):
    # Get current and next position data points
    current_point = position_data.iloc[i].values
    next_point = position_data.iloc[i + 1].values

    # Calculate number of points to insert (including the original points)
    num_inserts = int(np.floor(upsample_factor))

    # Calculate remaining samples that need to be handled
    remainder = (upsample_factor * (i + 1) - (i + 1)) / num_samples_48hz

    # Interpolate and add new points
    for j in range(num_inserts + 1):  # +1 to include the original point and interpolated points
        t = (j + 1) / (num_inserts + 1)
        new_point = current_point + t * (next_point - current_point)
        upsampled_position_data.append(new_point)

    # Add an extra interpolated point to handle remainder if needed
    if remainder > 0 and i == num_samples_48hz - 2:  # Only add extra point at the end
        t = 1.0
        new_point = current_point + t * (next_point - current_point)
        upsampled_position_data.append(new_point)

# Add the last point of the original data
upsampled_position_data.append(position_data.iloc[-1].values)

# Convert upsampled position data to DataFrame
upsampled_position_df = pd.DataFrame(upsampled_position_data, columns=position_data.columns)

# Ensure the number of rows matches the IMU data
if len(upsampled_position_df) != num_samples_160hz:
    # Adjust the number of rows if needed
    upsampled_position_df = upsampled_position_df.iloc[:num_samples_160hz]

# Merge the upsampled position data with IMU data
merged_data = pd.concat([imu_data.reset_index(drop=True), upsampled_position_df.reset_index(drop=True)], axis=1)

# Save the merged data to a new CSV file
merged_data.to_csv('evaluation_data2.csv', index=False, header=False)

print("Merged data saved to 'merged_data.csv'")
