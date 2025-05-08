import numpy as np
import pandas as pd

# Load the data from the text file
with open('DS1(prime)IPOPT.txt', 'r') as f:
    content = f.read()

# Extract the arrays
dvh_array = np.array([
    [ 7.64, 2.92, 19.62,  3.06, 12.58, 12.13,  3.37,  3.28, 0.00, 0.00, 0.00, 0.00,   0.00],  # Rectum
    [ 4.75, 0.50,  5.58,  4.04,  5.53,  5.49,  4.08,  4.06, 0.00, 0.00, 0.00, 0.00,   0.00],  # Penile bulb
    [ 5.37, 4.02, 48.79,  1.90, 16.07, 13.00,  2.35,  2.19, 0.09, 0.00, 0.00, 0.00,   0.00],  # Lymph Nodes
    [ 1.16, 0.81,  4.25,  0.24,  3.40,  2.91,  0.39,  0.33, 0.00, 0.00, 0.00, 0.00,   0.00],  # Rt femoral head
    [13.68, 1.21, 18.33,  7.60, 15.64, 14.90, 11.17,  9.37, 0.00, 0.00, 0.00, 0.0471, 24.87],  # prostate bed
    [12.37, 2.08, 22.33,  5.49, 15.54, 14.72,  8.12,  6.87, 0.00, 0.00, 0.00, 0.00,   0.00],  # PTV_68
    [ 5.60, 5.15,105.30,  1.45, 18.37, 13.93,  1.94,  1.80, 0.32, 0.14, 0.08, 0.00,   0.00],  # PTV_56
    [ 5.69, 3.95, 24.95,  1.02, 14.25, 13.59,  1.49,  1.28, 0.00, 0.00, 0.00, 0.00,   0.00],  # Bladder
    [ 1.17, 2.58,131.12,  0.00,  9.52,  4.88,  0.00,  0.00, 0.03, 0.02, 0.01, 0.00,   0.00],  # BODY
    [ 0.95, 0.59,  3.21,  0.08,  2.56,  2.24,  0.30,  0.24, 0.00, 0.00, 0.00, 0.00,   0.00],  # Lt femoral head
])

# Extract template positions (inputs)
templateX = np.array([float(x) for x in content.split('templateX:')[1].split('templateY:')[0].split()])
templateY = np.array([float(x) for x in content.split('templateY:')[1].split('templateZ:')[0].split()])
templateZ = np.array([float(x) for x in content.split('templateZ:')[1].split('seedPosX:')[0].split()])

# Extract seed positions (outputs)
seedPosX = np.array([float(x) for x in content.split('seedPosX:')[1].split('seedPosY:')[0].split()])
seedPosY = np.array([float(x) for x in content.split('seedPosY:')[1].split('seedPosZ:')[0].split()])
seedPosZ = np.array([float(x) for x in content.split('seedPosZ:')[1].split('objFunctionValue')[0].split()])

# Extract the objective function value (loss)
objFunctionValue = float(content.split('objFunctionValue :')[1].strip())

# Create a DataFrame
data = {
    'templateX': templateX,
    'templateY': templateY,
    'templateZ': templateZ,
    'seedPosX': seedPosX,
    'seedPosY': seedPosY,
    'seedPosZ': seedPosZ
}

df = pd.DataFrame(data)

# Add the objective function value as a constant column (same for all rows)
df['objFunctionValue'] = objFunctionValue

# Save to CSV if needed
df.to_csv('brachytherapy_dataset.csv', index=False)

# Display the first few rows
print(df.head())