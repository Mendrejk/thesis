import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob
import psutil
import keras
from keras import backend as K
import gc

class STFTDataset(Dataset):
    def __init__(self, high_quality_dir, low_quality_dir, transform=None, max_cache_size_gb=24):
        self.high_quality_files = sorted(glob.glob(os.path.join(high_quality_dir, '*_stft.npz')))
        self.low_quality_files = sorted(glob.glob(os.path.join(low_quality_dir, '*_stft.npz')))
        assert len(self.high_quality_files) == len(self.low_quality_files), "Mismatch in number of files"
        self.transform = transform
        self.cache = {}
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.current_cache_size = 0

    def __len__(self):
        return len(self.high_quality_files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        hq_file = self.high_quality_files[idx]
        lq_file = self.low_quality_files[idx]

        if idx not in self.cache:
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

            # Check if we have enough space to cache this item
            item_size = hq_stft.element_size() * hq_stft.nelement() + lq_stft.element_size() * lq_stft.nelement()
            if self.current_cache_size + item_size <= self.max_cache_size:
                self.cache[idx] = (lq_stft, hq_stft)
                self.current_cache_size += item_size
            else:
                print(f"Cache full. Using memory-mapped file for item {idx}")
                return lq_stft, hq_stft

        return self.cache[idx]


def prepare_data(high_quality_dir, low_quality_dir, batch_size=32, val_split=0.2):
    dataset = STFTDataset(high_quality_dir, low_quality_dir)

    # Split into train and validation sets
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def main():
    print(f"Keras backend: {K.backend()}")
    print(f"Keras version: {keras.__version__}")

    # Set directories
    high_quality_dir = "../data/converted/STFT"
    low_quality_dirs = [
        "../data/low_quality/STFT",
        "../data/ultra_low_quality/STFT",
        "../data/ultra_low_quality/STFT",
        "../data/vinyl_crackle/STFT"
    ]

    for low_quality_dir in low_quality_dirs:
        print(f"Preparing data for {low_quality_dir}")
        train_loader, val_loader = prepare_data(high_quality_dir, low_quality_dir)

        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")

        print("Loading first batch...")
        lq_batch, hq_batch = next(iter(train_loader))
        print(f"Low quality batch shape: {lq_batch.shape}")
        print(f"High quality batch shape: {hq_batch.shape}")

        print_memory_usage()

        print("\n")


if __name__ == "__main__":
    main()