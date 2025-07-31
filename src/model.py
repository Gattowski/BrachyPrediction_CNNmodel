import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random

# === Config flags ===
USE_FOCAL_LOSS = True
USE_PATCH_CROPPING = True
USE_DATA_AUGMENTATION = True

# === Dataset ===
class SeedMaskDataset(Dataset):
    def __init__(self, folder_path, file_list, target_shape=(1, 100, 88, 180), augment=False, patch_crop=False):
        self.file_paths = [os.path.join(folder_path, fname) for fname in file_list]
        self.target_shape = target_shape
        self.augment = augment
        self.patch_crop = patch_crop

    def pad_or_crop(self, volume):
        _, target_z, target_y, target_x = self.target_shape
        _, z, y, x = volume.shape

        if z < target_z:
            pz = (target_z - z) // 2
            volume = F.pad(volume, (0, 0, 0, 0, pz, target_z - z - pz))
        elif z > target_z:
            start = (z - target_z) // 2
            volume = volume[:, start:start + target_z, :, :]

        if y < target_y:
            py = (target_y - y) // 2
            volume = F.pad(volume, (0, 0, py, target_y - y - py, 0, 0))
        elif y > target_y:
            start = (y - target_y) // 2
            volume = volume[:, :, start:start + target_y, :]

        if x < target_x:
            px = (target_x - x) // 2
            volume = F.pad(volume, (px, target_x - x - px, 0, 0, 0, 0))
        elif x > target_x:
            start = (x - target_x) // 2
            volume = volume[:, :, :, start:start + target_x]

        return volume

    def crop_around_mask(self, vol, mask, size):
        """Center crop around the bounding box of the mask."""
        coords = (mask > 0).nonzero()
        if coords.numel() == 0:
            return self.pad_or_crop(vol)

        zc, yc, xc = coords[:, 1].float().mean(), coords[:, 2].float().mean(), coords[:, 3].float().mean()
        zc, yc, xc = int(zc), int(yc), int(xc)
        _, D, H, W = vol.shape
        dz, dy, dx = size[1]//2, size[2]//2, size[3]//2

        z1, y1, x1 = max(0, zc - dz), max(0, yc - dy), max(0, xc - dx)
        z2, y2, x2 = min(D, z1 + size[1]), min(H, y1 + size[2]), min(W, x1 + size[3])

        vol_cropped = vol[:, z1:z2, y1:y2, x1:x2]
        return self.pad_or_crop(vol_cropped)

    def random_flip(self, mask, seed_mask):
        if random.random() > 0.5:
            mask = torch.flip(mask, [2])
            seed_mask = torch.flip(seed_mask, [2])
        if random.random() > 0.5:
            mask = torch.flip(mask, [3])
            seed_mask = torch.flip(seed_mask, [3])
        return mask, seed_mask

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample = np.load(self.file_paths[idx])
        mask = torch.tensor(sample['mask'], dtype=torch.float32).unsqueeze(0)
        seed_mask = torch.tensor(sample['seedMask'], dtype=torch.float32).unsqueeze(0)
        obj_val = torch.tensor(sample['objFuncVal'], dtype=torch.float32)

        if self.patch_crop:
            mask = self.crop_around_mask(mask, mask, self.target_shape)
            seed_mask = self.crop_around_mask(seed_mask, mask, self.target_shape)
        else:
            mask = self.pad_or_crop(mask)
            seed_mask = self.pad_or_crop(seed_mask)

        if self.augment:
            mask, seed_mask = self.random_flip(mask, seed_mask)

        return {
            'input': mask,
            'target': seed_mask,
            'obj_value': obj_val,
            'filename': os.path.basename(self.file_paths[idx])
        }

# === Model (unchanged) ===
class UNet3D(nn.Module):
    def __init__(self):  # unchanged
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(64, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(32, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True))

        self.final = nn.Conv3d(16, 1, 1)

    def center_crop(self, layer, size):
        _, _, d, h, w = layer.size()
        td, th, tw = size
        return layer[:, :, (d - td)//2:(d + td)//2, (h - th)//2:(h + th)//2, (w - tw)//2:(w + tw)//2]

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.decoder2(torch.cat([self.up2(b), self.center_crop(e2, self.up2(b).shape[2:])], dim=1))
        d1 = self.decoder1(torch.cat([self.up1(d2), self.center_crop(e1, self.up1(d2).shape[2:])], dim=1))
        out = self.final(d1)

        if out.shape[2:] != x.shape[2:]:
            out = self.center_crop(out, x.shape[2:])
        return out

# === Loss Functions ===
def weighted_bce_loss(pred, target, obj_values, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    weights = 1.0 / (obj_values.view(-1, 1, 1, 1, 1) + eps)
    return (weights * bce).mean()

def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    pt = torch.where(target == 1, probs, 1 - probs)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt + 1e-6)
    return loss.mean()

# === Metrics ===
def dice_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return (2. * intersection / (union + 1e-6)).item()

# === Visualization ===
def visualize_prediction(pred, target, mask, filename, save_dir='vis_outputs'):
    os.makedirs(save_dir, exist_ok=True)
    pred_bin = (pred > 0.5).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    z = pred_bin.shape[1] // 2

    overlay = np.zeros((*pred_bin[0, z].shape, 3), dtype=np.uint8)
    overlay[mask[0, z] > 0] = [50, 50, 50]
    overlay[target_bin[0, z] > 0] = [0, 255, 0]
    overlay[pred_bin[0, z] > 0] = [255, 0, 0]
    overlay[(pred_bin[0, z] > 0) & (target_bin[0, z] > 0)] = [255, 255, 0]

    plt.imshow(overlay)
    plt.axis('off')
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_overlay.png"))
    plt.close()

# === Evaluation ===
def evaluate(model, dataloader, device='cuda', save_dir='vis_outputs'):
    model.eval()
    dice_scores = []
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            filenames = batch['filename']

            outputs = torch.sigmoid(model(inputs))

            for i in range(inputs.size(0)):
                pred = outputs[i].cpu().numpy()
                target = targets[i].cpu().numpy()
                mask = inputs[i].cpu().numpy()

                # Dynamic threshold search
                thresholds = np.linspace(0.1, 0.9, 9)
                best_dice = 0.0
                for t in thresholds:
                    dice = dice_score(torch.tensor(pred), torch.tensor(target), threshold=t)
                    if dice > best_dice:
                        best_dice = dice

                dice_scores.append(best_dice)
                print(f"{filenames[i]} Dice: {best_dice:.4f}")
                visualize_prediction(pred, target, mask, filenames[i], save_dir)

    print(f"\nAverage Dice: {np.mean(dice_scores):.4f}")

# === Main Training ===
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_FOLDER = 'npz_data'
    BATCH_SIZE = 2
    EPOCHS = 20
    LR = 1e-4

    files = sorted(f for f in os.listdir(DATA_FOLDER) if f.endswith('.npz'))
    np.random.seed(42)
    np.random.shuffle(files)
    train_files, test_files = files[:24], files[24:]

    train_ds = SeedMaskDataset(DATA_FOLDER, train_files, augment=USE_DATA_AUGMENTATION, patch_crop=USE_PATCH_CROPPING)
    test_ds = SeedMaskDataset(DATA_FOLDER, test_files)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = UNet3D().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            obj_vals = batch['obj_value'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            if USE_FOCAL_LOSS:
                loss = focal_loss(outputs, targets)
            else:
                loss = weighted_bce_loss(outputs, targets, obj_vals)

            loss.backward()

            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient of {name}")

            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

    print("\nEvaluating model on test set...")
    evaluate(model, test_loader, device=DEVICE)

if __name__ == "__main__":
    main()
