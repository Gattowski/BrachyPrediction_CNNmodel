import h5py
import numpy as np
import os

def dereference_cell_array(h5file, cell_array_ref):
    """
    Dereference a MATLAB cell array stored as object references in HDF5.
    Returns a list of numpy arrays or scalars.
    """
    result = []
    refs = np.array(cell_array_ref).flatten()
    for i, ref in enumerate(refs):
        # If ref is an ndarray with single element, extract that element
        if isinstance(ref, np.ndarray):
            ref = ref[0]
        print(f"Dereferencing cell {i}: {ref}")
        data = np.array(h5file[ref])
        # Convert to scalar if size==1
        if data.size == 1:
            data = data.item()
        result.append(data)
    return result

def extract_and_save(mat_file, output_folder):
    with h5py.File(mat_file, 'r') as f:
        mask_all = np.array(f['allMasks']).astype(np.uint8)         # (Z, Y, X, N)
        seedmask_all = np.array(f['allSeedMasks']).astype(np.uint8) # (Z, Y, X, N)

        # Dereference allObjFuncVals cell array
        allObjFuncVals = dereference_cell_array(f, f['allObjFuncVals'])
        print(f"allObjFuncVals type: {type(allObjFuncVals)}")
        print(f"allObjFuncVals length: {len(allObjFuncVals)}")

        total_samples = mask_all.shape[-1]
        print(f"Total samples to save: {total_samples}")

        for i in range(total_samples):
            mask = mask_all[:, :, :, i]
            seedmask = seedmask_all[:, :, :, i]

            objVal = allObjFuncVals[i]

            sample_data = {
                'mask': mask,             # input
                'seedMask': seedmask,     # binary target output
                'objFuncVal': objVal      # scalar objective function value
            }

            for k, v in sample_data.items():
                print(f"Sample {i+1} - {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")

            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            npz_name = f"{base_name}_sample{i+1}.npz"
            npz_path = os.path.join(output_folder, npz_name)

            np.savez_compressed(npz_path, **sample_data)
            print(f"Saved: {npz_path}")

# === Main ===
mat_folder = "matfiles"
output_folder = "npz_data"
os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(mat_folder)):
    if filename.endswith(".mat"):
        mat_path = os.path.join(mat_folder, filename)
        print(f"\nConverting: {mat_path}")
        extract_and_save(mat_path, output_folder)

print("\ndone.")
