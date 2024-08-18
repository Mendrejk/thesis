import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

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

        # Store original magnitudes for reconstruction
        clean_mag_original = clean_mag.copy()
        noisy_mag_original = noisy_mag.copy()

        # Normalize magnitude to [-1, 1] for model input
        clean_mag_norm = (clean_mag - np.min(clean_mag)) / (np.max(clean_mag) - np.min(clean_mag)) * 2 - 1
        noisy_mag_norm = (noisy_mag - np.min(noisy_mag)) / (np.max(noisy_mag) - np.min(noisy_mag)) * 2 - 1

        # Log data ranges
        logger.debug(f"Clean mag range: [{np.min(clean_mag_norm):.4f}, {np.max(clean_mag_norm):.4f}]")
        logger.debug(f"Noisy mag range: [{np.min(noisy_mag_norm):.4f}, {np.max(noisy_mag_norm):.4f}]")
        logger.debug(f"Clean phase range: [{np.min(clean_phase):.4f}, {np.max(clean_phase):.4f}]")
        logger.debug(f"Noisy phase range: [{np.min(noisy_phase):.4f}, {np.max(noisy_phase):.4f}]")

        clean_data_norm = np.stack([clean_mag_norm, clean_phase], axis=0)
        noisy_data_norm = np.stack([noisy_mag_norm, noisy_phase], axis=0)

        clean_data_original = np.stack([clean_mag_original, clean_phase], axis=0)
        noisy_data_original = np.stack([noisy_mag_original, noisy_phase], axis=0)

        return (torch.from_numpy(noisy_data_norm).float(),
                torch.from_numpy(clean_data_norm).float(),
                torch.from_numpy(noisy_data_original).float(),
                torch.from_numpy(clean_data_original).float())


def prepare_data(converted_dir, vinyl_crackle_dir, test_size=0.2, subset_fraction=1.0, **kwargs):
    converted_files = [os.path.join(converted_dir, f) for f in os.listdir(converted_dir) if f.endswith('_stft.npz')]
    vinyl_crackle_files = [os.path.join(vinyl_crackle_dir, f) for f in os.listdir(vinyl_crackle_dir) if
                           f.endswith('_stft.npz')]

    converted_files.sort()
    vinyl_crackle_files.sort()
    assert len(converted_files) == len(vinyl_crackle_files), "Mismatch in number of files"

    logger.info(f"Total number of files: {len(converted_files)}")

    # Apply subset if fraction is less than 1
    if subset_fraction < 1.0:
        num_samples = int(len(converted_files) * subset_fraction)
        converted_files = converted_files[:num_samples]
        vinyl_crackle_files = vinyl_crackle_files[:num_samples]
        logger.info(f"Using subset of {num_samples} files")

    train_converted, val_converted, train_vinyl, val_vinyl = train_test_split(
        converted_files, vinyl_crackle_files, test_size=test_size, random_state=42)

    logger.info(f"Training set size: {len(train_converted)}")
    logger.info(f"Validation set size: {len(val_converted)}")

    train_dataset = STFTDataset(train_converted, train_vinyl)
    val_dataset = STFTDataset(val_converted, val_vinyl)

    train_kwargs = kwargs.copy()
    train_kwargs['shuffle'] = True
    val_kwargs = kwargs.copy()
    val_kwargs['shuffle'] = False

    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    return train_loader, val_loader