import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch_directml
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting

#try:
    #DEVICE = torch_directml.device()
#except Exception as e:
    #print(f"DirectML device not found, falling back to CPU: {e}")
DEVICE = torch.device('cpu')

# === Dataset ===
class SeedPositionDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the .npz file on demand
        file_path = os.path.join(self.folder_path, self.files[idx])
        sample = np.load(file_path)

        mask = sample['mask']  # shape: (88, 180, 180)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim => (1, 88, 180, 180)

        seed_heatmap = sample['seedHeatmap']
        seed_heatmap = torch.tensor(seed_heatmap, dtype=torch.float32).unsqueeze(0)  # (1, 88, 180, 180)

        obj_value = torch.tensor(sample['objFuncVal'], dtype=torch.float32)

        filename = os.path.splitext(self.files[idx])[0]  # filename without extension

        return {
            'input': mask,
            'target': seed_heatmap,
            'obj_value': obj_value,
            'filename': filename,
        }




# === 3D U-Net ===
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        x = self.double_conv(x)
        
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Down part (encoding path)
        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up part (decoding path)
        rev_features = list(reversed(features))
        channels = features[-1] * 2
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose3d(channels, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
            channels = feature

        # Final conv layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        
        skip_connections = []

        for i, down in enumerate(self.downs):
            
            x = down(x)
            
            skip_connections.append(x)
            x = self.pool(x)
            

        
        x = self.bottleneck(x)
        

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            
            x = self.ups[idx](x)  # upconv
            

            skip_connection = skip_connections[idx // 2]
            

            # If needed, pad x to match skip_connection size due to odd shapes
            if x.shape != skip_connection.shape:
                diffD = skip_connection.shape[2] - x.shape[2]
                diffH = skip_connection.shape[3] - x.shape[3]
                diffW = skip_connection.shape[4] - x.shape[4]
                
                x = nn.functional.pad(x, [diffW // 2, diffW - diffW // 2,
                                          diffH // 2, diffH - diffH // 2,
                                          diffD // 2, diffD - diffD // 2])

            x = torch.cat((skip_connection, x), dim=1)
            

            x = self.ups[idx + 1](x)  # double conv
            

        final_out = self.final_conv(x)
        
        return final_out



# === Loss ===
criterion = nn.BCEWithLogitsLoss(reduction='none')

def custom_loss(pred, target, obj_val):
    """
    Weights BCE loss more heavily for samples whose obj_val is closer to the batch minimum.
    """
    loss = criterion(pred, target)  # (B, 1, D, H, W)

    min_obj = obj_val.min()  # scalar

    # Compute weights inversely proportional to distance from min_obj (add epsilon to avoid div by 0)
    eps = 1e-8
    weights = 1.0 / (obj_val - min_obj + eps)  # shape (B,)

    # Normalize weights to have mean = 1 to keep loss scale stable
    weights = weights / weights.mean()

    # Reshape weights for broadcasting to loss shape (B,1,1,1,1)
    weights = weights.view(-1, 1, 1, 1, 1)

    weighted_loss = loss * weights

    return weighted_loss.mean()

def extract_grid_seeds(heatmap, threshold=0.5, spacing=2):
    """
    Extract seed points from a 3D heatmap using thresholding and grid sampling.

    Args:
        heatmap (np.ndarray): 3D numpy array (D, H, W)
        threshold (float): threshold to consider a voxel a seed
        spacing (int): grid spacing to reduce number of seeds (e.g., 2)

    Returns:
        coords (list of tuples): List of (z, y, x) coordinates of detected seeds
    """
    mask = heatmap > threshold
    coords = []
    D, H, W = heatmap.shape

    for z in range(0, D, spacing):
        for y in range(0, H, spacing):
            for x in range(0, W, spacing):
                if mask[z, y, x]:
                    coords.append((z, y, x))
    return coords

def save_coords(coords, filepath):
    """
    Save coordinates to a txt file.

    Args:
        coords (list of tuples): (z,y,x) coordinates
        filepath (str): path to save txt file
    """
    with open(filepath, 'w') as f:
        for c in coords:
            f.write(f"{c[0]} {c[1]} {c[2]}\n")


def save_3d_plot(coords, title, filepath):
    """
    Save a 3D scatter plot of seed coordinates.

    Args:
        coords (list of tuples): (z,y,x) coordinates
        title (str): plot title
        filepath (str): path to save plot image
    """
    if len(coords) == 0:
        print(f"No coords to plot for {title}")
        return

    zs, ys, xs = zip(*coords)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='r', marker='o', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.savefig(filepath)
    plt.close(fig)

def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > threshold).astype(np.float32)
    intersection = np.sum(pred_bin * target_bin)
    total = np.sum(pred_bin) + np.sum(target_bin)
    dice = (2.0 * intersection + 1e-8) / (total + 1e-8)
    return dice

def save_overlay(pred_heatmap, mask, filename, save_path):
    mid_slice = pred_heatmap.shape[0] // 2
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(mask[mid_slice], cmap='gray', alpha=0.5)
    ax.imshow(pred_heatmap[mid_slice], cmap='hot', alpha=0.5)
    ax.set_title(f"{filename} - Mid Z Slice Overlay")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


# === Evaluation with saving predicted heatmap ===
def evaluate(model, dataloader, save_dir='./tests'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(DEVICE)  # shape: (B, 1, D, H, W)
            masks = inputs.cpu().squeeze(1).numpy()  # shape: (B, D, H, W)
            filenames = batch['filename']

            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().squeeze(1).numpy()  # shape: (B, D, H, W)

            batch_size = preds.shape[0]
            for i in range(batch_size):
                filename = filenames[i]
                pred_heatmap = preds[i]
                mask = masks[i]

                # Keep predictions only inside prostate region
                pred_heatmap *= mask  # zero out predictions outside prostate

                # Save predicted heatmap as .npy
                np.save(os.path.join(save_dir, f"{filename}_pred_heatmap.npy"), pred_heatmap)

                # Optional: Plot 3D scatter of nonzero values
                zyx = np.argwhere(pred_heatmap > 0.01)  # threshold for visibility
                if len(zyx) == 0:
                    print(f"No predicted seeds in {filename}")
                    continue

                zs, ys, xs = zyx[:, 0], zyx[:, 1], zyx[:, 2]
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xs, ys, zs, c='red', s=5)
                ax.set_title(f"Predicted Heatmap (within mask) - {filename}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Optionally compare to prostate bounding box
                dz, dy, dx = mask.shape
                ax.set_xlim([0, dx])
                ax.set_ylim([0, dy])
                ax.set_zlim([0, dz])

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{filename}_pred_heatmap_plot.png"))
                plt.close(fig)
            
                gt_heatmap = batch['target'].numpy()[i, 0]  # shape (D,H,W)
                dice = dice_score(pred_heatmap, gt_heatmap)
                print(f"{filename} Dice Score: {dice:.4f}")

                inside = np.sum(pred_heatmap * mask)
                outside = np.sum(pred_heatmap * (1 - mask))
                total = inside + outside
                inside_ratio = inside / total if total > 0 else 0
                print(f"{filename} → Inside ratio: {inside_ratio:.3f} (should be close to 1.0)")

                save_overlay(pred_heatmap, mask, filename, os.path.join(save_dir, f"{filename}_overlay.png"))
                num_seeds = np.sum(pred_heatmap > 0.5)
                print(f"{filename} → Candidate seed voxels: {num_seeds}")







# === Training ===
def train(model, train_loader, val_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            obj_val = batch['obj_value'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, targets, obj_val)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} train loss: {avg_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                obj_val = batch['obj_value'].to(DEVICE)
                outputs = model(inputs)
                loss = custom_loss(outputs, targets, obj_val)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} val loss: {val_loss:.4f}")

    print("Training done.")


# === Main ===
if __name__ == '__main__':
    folder_path = './npz_data'  # Change to your dataset folder
    dataset = SeedPositionDataset(folder_path)

    # Split dataset: first 33 train, last 8 test
    train_size = 33
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    model = UNet3D(in_channels=1, out_channels=1, features=[32, 64, 128, 256])

    train(model, train_loader, test_loader, epochs=10, lr=1e-3)

    # Save model weights
    torch.save(model.state_dict(), 'unet3d_model.pth')

    # Evaluate on test set and save outputs
    evaluate(model, test_loader, save_dir='./tests')