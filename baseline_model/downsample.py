import pandas as pd

# Load IMU data from CSV file
imu_data = pd.read_csv('imu_data.csv')

# Original and target row counts
original_rows = 18158
target_rows = 4000

# Calculate the downsample factor
downsample_factor = original_rows // target_rows

# Columns to downsample
columns_to_downsample = ['linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
                         'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z']

# Initialize a list to store downsampled data
downsampled_data = []

# Loop through the IMU data in chunks of downsample_factor
for start in range(13361, original_rows, downsample_factor):
    end = start + downsample_factor
    chunk = imu_data.iloc[start:end]
    
    # Calculate the mean of the specified columns in the chunk
    downsampled_chunk = chunk[columns_to_downsample].mean(axis=0)
    
    # Append the downsampled chunk to the downsampled data list
    downsampled_data.append(downsampled_chunk)

# Convert the list to a DataFrame
downsampled_imu_data = pd.DataFrame(downsampled_data)

# Rename columns to fit the specified format
downsampled_imu_data.columns = ['accel.linear.x', 'accel.linear.y', 'accel.linear.z',
                                'accel.angular.x', 'accel.angular.y', 'accel.angular.z']

# Save the downsampled data to a new CSV file
downsampled_imu_data.to_csv('downsampled_imu_data.csv', index=False)

print(f"Downsampled data saved to 'downsampled_imu_data.csv' with {len(downsampled_imu_data)} rows.")
