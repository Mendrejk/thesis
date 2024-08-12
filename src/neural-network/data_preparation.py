import os
import numpy as np
from keras import ops
from keras.utils import Sequence
from sklearn.model_selection import train_test_split


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

            clean_mag, clean_phase = np.abs(clean_stft), np.angle(clean_stft)
            noisy_mag, noisy_phase = np.abs(noisy_stft), np.angle(noisy_stft)

            clean_mag = (clean_mag - np.mean(clean_mag)) / np.std(clean_mag)
            noisy_mag = (noisy_mag - np.mean(noisy_mag)) / np.std(noisy_mag)

            clean_data = np.stack([clean_mag, clean_phase], axis=-1)
            noisy_data = np.stack([noisy_mag, noisy_phase], axis=-1)

            batch_clean.append(clean_data)
            batch_noisy.append(noisy_data)

        return ops.convert_to_tensor(np.array(batch_noisy)), ops.convert_to_tensor(np.array(batch_clean))


def prepare_data(converted_dir, vinyl_crackle_dir, test_size=0.2, batch_size=64):
    converted_files = [os.path.join(converted_dir, f) for f in os.listdir(converted_dir) if f.endswith('_stft.npz')]
    vinyl_crackle_files = [os.path.join(vinyl_crackle_dir, f) for f in os.listdir(vinyl_crackle_dir) if
                           f.endswith('_stft.npz')]

    converted_files.sort()
    vinyl_crackle_files.sort()
    assert len(converted_files) == len(vinyl_crackle_files), "Mismatch in number of files"

    train_converted, val_converted, train_vinyl, val_vinyl = train_test_split(
        converted_files, vinyl_crackle_files, test_size=test_size, random_state=42)

    train_dataset = STFTDataset(train_converted, train_vinyl, batch_size)
    val_dataset = STFTDataset(val_converted, val_vinyl, batch_size)

    return train_dataset, val_dataset
