import os

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np
import torch
from keras import ops
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class STFTDataset(Sequence):
    def __init__(self, clean_files, noisy_files, batch_size=32):
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.clean_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_clean_files = self.clean_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_noisy_files = self.noisy_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_clean = []
        batch_noisy = []

        for clean_file, noisy_file in zip(batch_clean_files, batch_noisy_files):
            clean_stft = np.load(clean_file)['stft']
            noisy_stft = np.load(noisy_file)['stft']

            # Convert complex STFTs to magnitude and phase
            clean_mag, clean_phase = np.abs(clean_stft), np.angle(clean_stft)
            noisy_mag, noisy_phase = np.abs(noisy_stft), np.angle(noisy_stft)

            # Normalize magnitudes
            clean_mag = (clean_mag - np.mean(clean_mag)) / np.std(clean_mag)
            noisy_mag = (noisy_mag - np.mean(noisy_mag)) / np.std(noisy_mag)

            # Stack magnitude and phase
            clean_data = np.stack([clean_mag, clean_phase], axis=-1)
            noisy_data = np.stack([noisy_mag, noisy_phase], axis=-1)

            batch_clean.append(clean_data)
            batch_noisy.append(noisy_data)

        return ops.convert_to_tensor(np.array(batch_noisy)), ops.convert_to_tensor(np.array(batch_clean))


def prepare_data(converted_dir, vinyl_crackle_dir, test_size=0.2, batch_size=64):
    # Get file lists
    converted_files = [os.path.join(converted_dir, f) for f in os.listdir(converted_dir) if f.endswith('_stft.npz')]
    vinyl_crackle_files = [os.path.join(vinyl_crackle_dir, f) for f in os.listdir(vinyl_crackle_dir) if
                           f.endswith('_stft.npz')]

    # Ensure we have matching pairs
    converted_files.sort()
    vinyl_crackle_files.sort()
    assert len(converted_files) == len(vinyl_crackle_files), "Mismatch in number of files"

    # Split into train and validation sets
    train_converted, val_converted, train_vinyl, val_vinyl = train_test_split(
        converted_files, vinyl_crackle_files, test_size=test_size, random_state=42)

    # Create datasets
    train_dataset = STFTDataset(train_converted, train_vinyl, batch_size)
    val_dataset = STFTDataset(val_converted, val_vinyl, batch_size)

    return train_dataset, val_dataset


def estimate_memory_usage(batch_size, stft_shape=(1025, 862)):
    # Each STFT is complex64 (8 bytes per element)
    single_stft_memory = np.prod(stft_shape) * 8 * 2  # *2 for real and imaginary parts

    # We have noisy and clean samples, each with magnitude and phase
    total_memory_per_batch = single_stft_memory * batch_size * 2 * 2

    # Convert to GB
    total_memory_gb = total_memory_per_batch / (1024 ** 3)

    return total_memory_gb


if __name__ == "__main__":
    converted_dir = "../data/converted/stft_segments"
    vinyl_crackle_dir = "../data/vinyl_crackle/stft_segments"

    # Estimate memory usage for different batch sizes
    for batch_size in [32, 64, 128, 256]:
        memory_usage = estimate_memory_usage(batch_size)
        print(f"Estimated memory usage for batch size {batch_size}: {memory_usage:.2f} GB")

    # Choose a batch size based on the estimates and your GPU memory
    batch_size = 16  # Adjust this based on the estimates and leaving some headroom

    train_dataset, val_dataset = prepare_data(converted_dir, vinyl_crackle_dir, batch_size=batch_size)

    print(f"Number of training batches: {len(train_dataset)}")
    print(f"Number of validation batches: {len(val_dataset)}")

    # Test loading a batch
    noisy_batch, clean_batch = train_dataset[0]
    print(f"Noisy batch shape: {noisy_batch.shape}")
    print(f"Clean batch shape: {clean_batch.shape}")