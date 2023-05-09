from netCDF4 import Dataset
import pandas as pd
import numpy as np

def month_to_date(months, start_date):
    years, months_remaining = divmod(months, 12)
    return start_date + pd.DateOffset(years=years, months=months_remaining)

# Read the NetCDF file
nc_file = Dataset(r'D:/Final_sem_thesis/data/data_sst.nc')

z = nc_file.variables['zlev']
print(z)

'''
# Accessing the variables
time_data = nc_file.variables['T'][:]
x_data = nc_file.variables['X'][:]
y_data = nc_file.variables['Y'][:]
sst_data = nc_file.variables['sst'][:]

# Round the 'months since' values to the nearest integer
time_data = np.round(time_data).astype(int)

# Convert the time data to a pandas datetime object
time_units = nc_file.variables['T'].units
reference_date = pd.to_datetime(time_units.split('since')[1].strip())
time_data_datetime = [month_to_date(months, reference_date) for months in time_data]

# Create an empty DataFrame to store the final result
sst_df = pd.DataFrame(columns=['time', 'latitude', 'longitude', 'sst'])

# Iterate through the time dimension and create a DataFrame for each time step
for i, time_step in enumerate(time_data_datetime):
    sst_data_flat = sst_data[i].flatten()
    index = pd.MultiIndex.from_product([[time_step], y_data, x_data], names=['time', 'latitude', 'longitude'])
    temp_df = pd.DataFrame(sst_data_flat, index=index, columns=['sst']).reset_index()
    sst_df = sst_df.append(temp_df)

# Save the DataFrame as a CSV file
sst_df.to_csv(r'D:/Final_sem_thesis/data/data_sst_all_variables.csv', index=False)
'''