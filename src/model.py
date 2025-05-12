import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Dataset class
class SeedObjectiveDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        data = np.load(file_path)

        # Input: templateX, templateY, templateZ only
        input_vector = np.concatenate([
            data['templateX'],
            data['templateY'],
            data['templateZ']
        ])

        # Target: seedPosX, seedPosY, seedPosZ
        target_vector = np.stack([
            data['seedPosX'],
            data['seedPosY'],
            data['seedPosZ']
        ], axis=0)  # Shape: (3, 624)

        obj_value = data['objFunctionValue'].item()  # Scalar

        return {
            'input': torch.tensor(input_vector, dtype=torch.float32),
            'target': torch.tensor(target_vector, dtype=torch.float32),
            'obj_value': torch.tensor(obj_value, dtype=torch.float32)
        }

# Model class
class ObjectiveRegressor(nn.Module):
    def __init__(self, input_size, output_size=3, seq_length=624):
        super(ObjectiveRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size * seq_length)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 3, 624)

# Custom loss
def custom_loss(predictions, targets, obj_values):
    mse = (predictions - targets) ** 2
    loss_per_sample = mse.mean(dim=[1, 2])
    weights = 1.0 / (obj_values + 1e-8)
    weighted_loss = loss_per_sample * weights
    return weighted_loss.mean()

# Absolute difference metrics
def evaluate_absolute_difference(model, dataset):
    model.eval()
    print("\n--- Evaluation on last 10 samples ---")
    with torch.no_grad():
        abs_diffs = []
        for i in range(len(dataset) - 10, len(dataset)):
            sample = dataset[i]
            input_tensor = sample['input'].unsqueeze(0)  # Add batch dimension
            true_output = sample['target']
            pred_output = model(input_tensor)[0]  # Remove batch dimension

            abs_diff = torch.abs(pred_output - true_output)  # Shape: (3, 624)
            abs_diffs.append(abs_diff)

            mean_diff_per_axis = abs_diff.mean(dim=1)  # Mean over 624 per axis
            print(f"Sample {i}: Mean Abs Diff - X: {mean_diff_per_axis[0]:.4f}, Y: {mean_diff_per_axis[1]:.4f}, Z: {mean_diff_per_axis[2]:.4f}")

        overall_diff = torch.stack(abs_diffs).mean(dim=[0, 2])  # Mean over samples and seeds
        print(f"\nOverall Mean Abs Difference - X: {overall_diff[0]:.4f}, Y: {overall_diff[1]:.4f}, Z: {overall_diff[2]:.4f}")

# Main
def main():
    folder_path = './datasets'
    dataset = SeedObjectiveDataset(folder_path)

    # Split: first 40 for training, last 10 for evaluation
    train_dataset = torch.utils.data.Subset(dataset, list(range(40)))
    eval_dataset = torch.utils.data.Subset(dataset, list(range(40, 50)))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    input_size = len(dataset[0]['input'])
    model = ObjectiveRegressor(input_size=input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_val_loss = float('inf')
    best_model_state = None

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['input']
            targets = batch['target']
            obj_values = batch['obj_value']

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, targets, obj_values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            best_model_state = model.state_dict()
            print(f"New best model saved at epoch {epoch+1} with loss: {best_val_loss:.4f}")

    # Save final model
    if best_model_state:
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', 'seedPredictionModel.pth')
        torch.save(best_model_state, model_path)
        print(f"\nBest model saved to: {model_path}")

    # Load best model and evaluate
    model.load_state_dict(torch.load(model_path))
    evaluate_absolute_difference(model, dataset)

if __name__ == '__main__':
    main()
