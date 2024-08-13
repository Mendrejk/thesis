import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class STFTDataset(Dataset):
    def __init__(self, clean_files, noisy_files):
        self.clean_files = clean_files
        self.noisy_files = noisy_files

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_file = self.clean_files[idx]
        noisy_file = self.noisy_files[idx]

        clean_stft = np.load(clean_file)['stft']
        noisy_stft = np.load(noisy_file)['stft']

        clean_mag, clean_phase = np.abs(clean_stft), np.angle(clean_stft)
        noisy_mag, noisy_phase = np.abs(noisy_stft), np.angle(noisy_stft)

        clean_mag = (clean_mag - np.mean(clean_mag)) / np.std(clean_mag)
        noisy_mag = (noisy_mag - np.mean(noisy_mag)) / np.std(noisy_mag)

        clean_data = np.stack([clean_mag, clean_phase], axis=0)
        noisy_data = np.stack([noisy_mag, noisy_phase], axis=0)

        return torch.from_numpy(noisy_data).float(), torch.from_numpy(clean_data).float()


def prepare_data(converted_dir, vinyl_crackle_dir, test_size=0.2, **kwargs):
    converted_files = [os.path.join(converted_dir, f) for f in os.listdir(converted_dir) if f.endswith('_stft.npz')]
    vinyl_crackle_files = [os.path.join(vinyl_crackle_dir, f) for f in os.listdir(vinyl_crackle_dir) if
                           f.endswith('_stft.npz')]

    converted_files.sort()
    vinyl_crackle_files.sort()
    assert len(converted_files) == len(vinyl_crackle_files), "Mismatch in number of files"

    train_converted, val_converted, train_vinyl, val_vinyl = train_test_split(
        converted_files, vinyl_crackle_files, test_size=test_size, random_state=42)

    train_dataset = STFTDataset(train_converted, train_vinyl)
    val_dataset = STFTDataset(val_converted, val_vinyl)

    train_kwargs = kwargs.copy()
    train_kwargs['shuffle'] = True
    val_kwargs = kwargs.copy()
    val_kwargs['shuffle'] = False

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    return train_loader, val_loader
