import h5py
import numpy as np
import os

def dereference_cell_array(h5file, cell_array_ref):
    result = []
    refs = np.array(cell_array_ref).flatten()  # convert to ndarray and flatten
    for i, ref in enumerate(refs):
        if isinstance(ref, np.ndarray):
            ref = ref[0]
        print(f"Dereferencing cell {i}: {ref}")
        data = np.array(h5file[ref])
        result.append(data)
    return result


def extract_and_save(mat_file, output_folder):
    with h5py.File(mat_file, 'r') as f:
        mask_all = np.array(f['allMasks']).astype(np.uint8)  # shape (N, 90, 183, 183)
        
        # Dereference all cell arrays
        templateX = dereference_cell_array(f, f['templateX'])
        templateY = dereference_cell_array(f, f['templateY'])
        allObjFuncVals = dereference_cell_array(f, f['allObjFuncVals'])
        seedX_list = dereference_cell_array(f, f['allSeedPointsX'])
        seedY_list = dereference_cell_array(f, f['allSeedPointsY'])
        seedZ_list = dereference_cell_array(f, f['allSeedPointsZ'])

        total_samples = len(seedX_list)
        print(f"Total samples to save: {total_samples}")

        for i in range(total_samples):
            sample_data = {
                'mask': mask_all[i],                      # ndarray (90,183,183)
                'templateX': templateX[i].squeeze(),     # dereferenced ndarray or scalar
                'templateY': templateY[i].squeeze(),
                'objFuncVal': allObjFuncVals[i].squeeze(),
                'seedPosX': seedX_list[i],
                'seedPosY': seedY_list[i],
                'seedPosZ': seedZ_list[i]
            }

            # Debug print of each sample component's type and shape
            for k, v in sample_data.items():
                print(f"Sample {i+1} - {k}: type={type(v)}, shape={getattr(v, 'shape', None)}")

            base_name = os.path.splitext(os.path.basename(mat_file))[0]
            npz_name = f"{base_name}_sample{i+1}.npz"
            npz_path = os.path.join(output_folder, npz_name)

            np.savez_compressed(npz_path, **sample_data)
            print(f"✅ Saved: {npz_path}")

# === Main ===
mat_folder = "matfiles"
output_folder = "npz_data"
os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(mat_folder)):
    if filename.endswith(".mat"):
        mat_path = os.path.join(mat_folder, filename)
        print(f"\n🔄 Converting: {mat_path}")
        extract_and_save(mat_path, output_folder)

print("\n✅ All done.")
