import numpy as np

# Load your .npz file
data = np.load('datasets/DS1.npz')

# List all keys
print("Keys in .npz file:", data.files)

# Check shapes
print("templateX shape:", data['templateX'].shape)
print("templateY shape:", data['templateY'].shape)
print("templateZ shape:", data['templateZ'].shape)

print("seedPosX shape:", data['seedPosX'].shape)
print("seedPosY shape:", data['seedPosY'].shape)
print("seedPosZ shape:", data['seedPosZ'].shape)

print("objFunctionValue shape:", data['objFunctionValue'].shape)
