import torch
from torch.utils.data import Dataset
import ast

class SeedPlacementDataset(Dataset):
    def __init__(self, text_file_path):
        with open(text_file_path, 'r') as file:
            data = file.read()

        # Parse with AST for safety and structure
        parsed_data = ast.literal_eval(data)

        # Extract relevant parts
        coords_x = parsed_data['template.coords']['x']
        coords_y = parsed_data['template.coords']['y']
        coords_z = parsed_data['template.coords']['z']

        seed_x = parsed_data['solution.seedPositions']['x']
        seed_y = parsed_data['solution.seedPositions']['y']
        seed_z = parsed_data['solution.seedPositions']['z']

        self.obj_function_value = parsed_data['solution.objFunctionValue']

        # Ensure all arrays are of same length
        assert len(coords_x) == len(coords_y) == len(coords_z), "Coordinate arrays length mismatch"
        assert len(seed_x) == len(seed_y) == len(seed_z), "Seed position arrays length mismatch"

        self.inputs = torch.tensor(list(zip(coords_x, coords_y, coords_z)), dtype=torch.float32)
        self.targets = torch.tensor(list(zip(seed_x, seed_y, seed_z)), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],        # [x, y, z] of STF template
            'target': self.targets[idx],      # [seedPosX, seedPosY, seedPosZ]
            'loss': torch.tensor(self.obj_function_value, dtype=torch.float32)  # single value
        }

# Example usage
dataset = SeedPlacementDataset('/mnt/data/3191cbef-0c24-4ef6-8c2d-85f4cbd2b20a')
print(dataset[0])
