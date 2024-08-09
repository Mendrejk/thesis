import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import keras
from keras import backend as K
import gc

class STFTDataset(Dataset):
    def __init__(self, high_quality_dir, low_quality_dir, transform=None):
        self.high_quality_files = sorted(glob.glob(os.path.join(high_quality_dir, '*_stft.npz')))
        self.low_quality_files = sorted(glob.glob(os.path.join(low_quality_dir, '*_stft.npz')))
        assert len(self.high_quality_files) == len(self.low_quality_files), "Mismatch in number of files"
        self.transform = transform

    def __len__(self):
        return len(self.high_quality_files)

    def __getitem__(self, idx):
        hq_file = self.high_quality_files[idx]
        lq_file = self.low_quality_files[idx]

        hq_data = np.load(hq_file)
        lq_data = np.load(lq_file)

        hq_stft = hq_data['stft']
        lq_stft = lq_data['stft']

        # Handle complex data
        hq_stft = np.stack((hq_stft.real, hq_stft.imag), axis=-1)
        lq_stft = np.stack((lq_stft.real, lq_stft.imag), axis=-1)

        # Convert to PyTorch tensors
        hq_stft = torch.from_numpy(hq_stft).float()
        lq_stft = torch.from_numpy(lq_stft).float()

        # Ensure the tensors have the shape (channels, height, width, 2)
        hq_stft = hq_stft.permute(2, 0, 1, 3)
        lq_stft = lq_stft.permute(2, 0, 1, 3)

        if self.transform:
            hq_stft = self.transform(hq_stft)
            lq_stft = self.transform(lq_stft)

        return lq_stft, hq_stft


def prepare_data(high_quality_dir, low_quality_dir, batch_size=32, val_split=0.2):
    dataset = STFTDataset(high_quality_dir, low_quality_dir)

    # Split into train and validation sets
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_split, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def main():
    print(f"Keras backend: {K.backend()}")
    print(f"Keras version: {keras.__version__}")

    # Set directories
    high_quality_dir = "../data/converted/STFT"
    low_quality_dirs = [
        "../data/low_quality/STFT",
        "../data/no_noise_ultra_low_quality/STFT",
        "../data/ultra_low_quality/STFT",
        "../data/vinyl_crackle/STFT"
    ]

    # Prepare data for each low quality directory
    for low_quality_dir in low_quality_dirs:
        print(f"Preparing data for {low_quality_dir}")
        train_loader, val_loader = prepare_data(high_quality_dir, low_quality_dir)

        # Print some information about the data
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")

        # Check a single batch
        lq_batch, hq_batch = next(iter(train_loader))
        print(f"Low quality batch shape: {lq_batch.shape}")
        print(f"High quality batch shape: {hq_batch.shape}")

        print("\n")


if __name__ == "__main__":
    main()