import h5py
import numpy as np
import os

def dereference_cell_array(h5file, cell_array_ref):
    result = []
    refs = np.array(cell_array_ref).flatten()
    for i, ref in enumerate(refs):
        if isinstance(ref, np.ndarray):
            ref = ref[0]
        print(f"Dereferencing cell {i}: {ref}")
        data = np.array(h5file[ref])
        result.append(data)
    return result

def extract_and_save(mat_file, output_folder):
    with h5py.File(mat_file, 'r') as f:
        mask_all = np.array(f['allMasks']).astype(np.uint8)         # (Z, Y, X, N)
        seedmask_all = np.array(f['allSeedMasks']).astype(np.uint8) # (Z, Y, X, N)

        allObjFuncVals = dereference_cell_array(f, f['allObjFuncVals'])

        total_samples = mask_all.shape[-1]
        n_vals = len(allObjFuncVals)
        loop_len = min(total_samples, n_vals)

        print(f"Total samples in mask_all: {total_samples}")
        print(f"Length of allObjFuncVals: {n_vals}")
        print(f"Looping over {loop_len} samples.")

        for i in range(loop_len):
            mask = mask_all[:, :, :, i]
            seedmask = seedmask_all[:, :, :, i]

            val = allObjFuncVals[i]
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    val = val.item()
                else:
                    val = val.flatten()[0]
            elif isinstance(val, list):
                if len(val) == 1:
                    val = val[0]

            objVal = val

            print(f"Sample {i+1} objFuncVal: {objVal} (type: {type(objVal)})")

            sample_data = {
                'mask': mask,
                'seedMask': seedmask,
                'objFuncVal': objVal
            }

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
