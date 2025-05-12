import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# Dataset class
class SeedObjectiveDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

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

        # Target: seedPosX, seedPosY, seedPosZ (each has 624 values)
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

# Custom MSE loss weighted by inverse of objFunctionValue
def custom_loss(predictions, targets, obj_values):
    mse = (predictions - targets) ** 2
    loss_per_sample = mse.mean(dim=[1, 2])  # Mean over (3, 624) for each sample
    weights = 1.0 / (obj_values + 1e-8)      # Inverse weighting; small epsilon to avoid division by 0
    weighted_loss = loss_per_sample * weights
    return weighted_loss.mean()             # Average over batch

# Training script
def main():
    folder_path = './datasets'

    dataset = SeedObjectiveDataset(folder_path)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    input_size = len(dataset[0]['input'])
    model = ObjectiveRegressor(input_size=input_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_val_loss = float('inf')
    best_model = None

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

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input']
                targets = batch['target']
                obj_values = batch['obj_value']
                outputs = model(inputs)
                loss = custom_loss(outputs, targets, obj_values)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            print(f"New best model found with validation loss: {best_val_loss:.4f}")

    if best_model is not None:
        torch.save(best_model, 'seedPredictionModel.pth')
        print(f"Best model saved. Final best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
