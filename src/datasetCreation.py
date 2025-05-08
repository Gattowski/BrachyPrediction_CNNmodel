import os
import numpy as np


input_folder = 'textFileData'
output_folder = 'datasets'
os.makedirs(output_folder, exist_ok=True)

# Required keys and their expected order
keys = [
    #'dvh_array',
    'templateX:',
    'templateY:',
    'templateZ:',
    'seedPosX:',
    'seedPosY:',
    'seedPosZ:',
    'objFunctionValue:'
]

for filename in os.listdir(input_folder):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(input_folder, filename)
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Ensure all required keys exist
        for key in keys:
            if key not in content:
                raise ValueError(f"Missing key '{key}' in file {filename}")

        # Safely extract sections
        #dvh_raw      = content.split('dvh_array:')[1].split('templateX:')[0].strip()
        templateX_raw = content.split('templateX:')[1].split('templateY:')[0]
        templateY_raw = content.split('templateY:')[1].split('templateZ:')[0]
        templateZ_raw = content.split('templateZ:')[1].split('seedPosX:')[0]
        seedPosX_raw  = content.split('seedPosX:')[1].split('seedPosY:')[0]
        seedPosY_raw  = content.split('seedPosY:')[1].split('seedPosZ:')[0]
        seedPosZ_raw  = content.split('seedPosZ:')[1].split('objFunctionValue')[0]
        obj_val_raw   = content.split('objFunctionValue')[1]

        # Parse arrays
        #dvh_array = np.array(ast.literal_eval(dvh_raw))
        templateX = np.array([float(x) for x in templateX_raw.split()])
        templateY = np.array([float(x) for x in templateY_raw.split()])
        templateZ = np.array([float(x) for x in templateZ_raw.split()])
        seedPosX = np.array([float(x) for x in seedPosX_raw.split()])
        seedPosY = np.array([float(x) for x in seedPosY_raw.split()])
        seedPosZ = np.array([float(x) for x in seedPosZ_raw.split()])
        objFunctionValue = float(obj_val_raw.strip().replace(":", "").strip())

        # Save .npz
        out_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.npz')
        np.savez_compressed(
            out_path,
            #dvh_array=dvh_array,
            templateX=templateX,
            templateY=templateY,
            templateZ=templateZ,
            seedPosX=seedPosX,
            seedPosY=seedPosY,
            seedPosZ=seedPosZ,
            objFunctionValue=objFunctionValue
        )
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
