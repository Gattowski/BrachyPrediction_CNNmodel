import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIXED_Z_VALUE = 58.8712

# Dataset
class SeedGridDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
        self.seed_grid_positions = self._load_fixed_grid()

    def _load_fixed_grid(self):
        sample_path = os.path.join(self.folder_path, self.files[0])
        data = np.load(sample_path)
        return np.stack([
            data['seedPosX'], 
            data['seedPosY'], 
            data['seedPosZ']
        ], axis=1)  # shape (624, 3)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = np.load(file_path)
        
        templateX = data['templateX']
        templateY = data['templateY']
        fixedZ = np.full_like(templateX, FIXED_Z_VALUE)

        input_vector = np.concatenate([templateX, templateY, fixedZ])  # shape (312,)

        seed_mask = ((data['seedPosX'] != 0) | 
                     (data['seedPosY'] != 0) | 
                     (data['seedPosZ'] != 0)).astype(np.float32)  # shape (624,)
        
        obj_value = data['objFunctionValue'].item()

        return {
            'input': torch.tensor(input_vector, dtype=torch.float32),
            'target': torch.tensor(seed_mask, dtype=torch.float32),
            'obj_value': torch.tensor(obj_value, dtype=torch.float32),
        }

# Model
class SeedSelectorModel(nn.Module):
    def __init__(self, input_size=312, output_size=624):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))  # shape (batch, 624)

# Weighted BCE Loss
def weighted_bce_loss(predictions, targets, obj_values):
    loss_fn = nn.BCELoss(reduction='none')
    losses = loss_fn(predictions, targets)  # shape (batch, 624)
    weights = 1.0 / (obj_values.view(-1, 1) + 1e-8)
    weighted_losses = losses * weights
    return weighted_losses.mean()

# Visualize seeds in 3D
def visualize_3d_seeds(seed_coords, mask, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    active = mask > 0.5
    ax.scatter(seed_coords[~active, 0], seed_coords[~active, 1], seed_coords[~active, 2], c='gray', alpha=0.2, label='Off Seeds')
    ax.scatter(seed_coords[active, 0], seed_coords[active, 1], seed_coords[active, 2], c='red', label='On Seeds')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Predicted Active Seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Save ON-seed coordinates
def save_active_coords(seed_coords, mask, output_path):
    active = mask > 0.5
    active_coords = seed_coords[active]
    np.savetxt(output_path, active_coords, fmt="%.6f", header="X Y Z", comments='')

# Evaluation function
def evaluate(model, dataset, grid_positions, save_dir='./tests'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0)
            pred_mask = model(input_tensor)[0].cpu().numpy()

            # Save coordinates
            coord_path = os.path.join(save_dir, f'seed_coords_{i+1:02d}.txt')
            save_active_coords(grid_positions, pred_mask, coord_path)

            # Save 3D visualization
            fig_path = os.path.join(save_dir, f'seed_plot_{i+1:02d}.png')
            visualize_3d_seeds(grid_positions, pred_mask, fig_path)

            print(f"Sample {i+1}: saved coordinates + plot.")

# Main
def main():
    folder = './datasets'
    dataset = SeedGridDataset(folder)
    grid_positions = dataset.seed_grid_positions

    train_dataset = torch.utils.data.Subset(dataset, list(range(40)))
    eval_dataset = torch.utils.data.Subset(dataset, list(range(40, 50)))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = SeedSelectorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    best_state = None

    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input'])
            loss = weighted_bce_loss(outputs, batch['target'], batch['obj_value'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            print(f"New best model at epoch {epoch+1}.")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_seed_selector.pth'
    torch.save(best_state, model_path)
    print(f"\nBest model saved to {model_path}")

    model.load_state_dict(torch.load(model_path))
    evaluate(model, eval_dataset, grid_positions)

if __name__ == '__main__':
    main()
