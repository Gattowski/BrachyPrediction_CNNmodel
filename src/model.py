import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# === Dataset ===
class SeedPositionDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.folder_path, self.files[idx]))
        mask = data['mask']  # shape (D, H, W)
        prostate_binary = (mask > 0).astype(np.float32)
        seedX, seedY, seedZ = data['seedPosX'], data['seedPosY'], data['seedPosZ']
        obj_val = float(np.min(data['objFuncVal']))

        heatmap = np.zeros_like(mask, dtype=np.float32)
        for x, y, z in zip(seedX, seedY, seedZ):
            ix = int(round(float(np.squeeze(x))))
            iy = int(round(float(np.squeeze(y))))
            iz = int(round(float(np.squeeze(z))))
            if 0 <= iz < heatmap.shape[0] and 0 <= iy < heatmap.shape[1] and 0 <= ix < heatmap.shape[2]:
                heatmap[iz, iy, ix] = 1.0

            # === CROP to make shape (88, 180, 180) ===
            crop_d, crop_h, crop_w = 88, 180, 180
            prostate_binary = prostate_binary[:crop_d, :crop_h, :crop_w]
            heatmap = heatmap[:crop_d, :crop_h, :crop_w]

            assert prostate_binary.shape == (88, 180, 180), f"Input shape mismatch: {prostate_binary.shape}"
            assert heatmap.shape == (88, 180, 180), f"Target shape mismatch: {heatmap.shape}"

        return {
        'input': torch.tensor(prostate_binary[None], dtype=torch.float32),  # shape (1, D, H, W)
        'target': torch.tensor(heatmap[None], dtype=torch.float32),         # shape (1, D, H, W)
        'obj_value': torch.tensor(obj_val, dtype=torch.float32),
        'filename': self.files[idx]
        }

# === Model ===
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv3d(8, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === Loss ===
criterion = nn.BCEWithLogitsLoss(reduction='none')

def custom_loss(pred, target, obj_val):
    loss = criterion(pred, target)
    weight = 1.0 / (obj_val.view(-1, 1, 1, 1, 1) + 1e-8)
    return (loss * weight).mean()

# === Post-processing ===
def extract_grid_seeds(heatmap, threshold=0.5, spacing=2):
    binary = (heatmap > threshold).astype(np.uint8)
    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return np.empty((0, 3))
    grid = []
    for pt in coords:
        pt = tuple(pt)
        if all(np.linalg.norm(np.array(pt) - np.array(g)) >= spacing for g in grid):
            grid.append(pt)
    return np.array(grid)[:, [2, 1, 0]]

# === Save utilities ===
def save_3d_plot(coords, title, path):
    if coords.shape[0] == 0:
        print(f"{title}: No seed coordinates to plot.")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='red')
    ax.set_title(title)
    plt.savefig(path)
    plt.close()

def save_coords(coords, path):
    np.savetxt(path, coords, fmt='%.3f', header='X Y Z', comments='')

# === Evaluation ===
def evaluate(model, dataloader, save_dir='./tests'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(DEVICE)
            filenames = batch['filename']
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().squeeze(1).numpy()

            for i in range(len(filenames)):
                filename = filenames[i]
                heatmap = preds[i]
                coords = extract_grid_seeds(heatmap, threshold=0.5, spacing=2)
                save_coords(coords, os.path.join(save_dir, f"{filename}_coords.txt"))
                save_3d_plot(coords, filename, os.path.join(save_dir, f"{filename}_plot.png"))

# === Main ===
def main():
    data_folder = './npz_data'
    dataset = SeedPositionDataset(data_folder)
    train_data = torch.utils.data.Subset(dataset, range(33))
    test_data = torch.utils.data.Subset(dataset, range(34, 41))

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4)

    model = Simple3DCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float('inf')
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            obj_vals = batch['obj_value'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, targets, obj_vals)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model at epoch {epoch+1}")

    print("Evaluating best model...")
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate(model, test_loader)

if __name__ == '__main__':
    main()
