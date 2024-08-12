import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def check_stft_sample_rate(file_path):
    try:
        with np.load(file_path) as data:
            return file_path, data['sr']
    except Exception as e:
        return file_path, f"Error: {str(e)}"


def process_directory(directory):
    stft_dir = os.path.join(directory, 'stft_segments')
    if not os.path.exists(stft_dir):
        print(f"Warning: STFT directory not found: {stft_dir}")
        return []

    files = [os.path.join(stft_dir, f) for f in os.listdir(stft_dir) if f.endswith('_stft.npz')]

    anomalies = []
    expected_sr = 44100 if 'converted' in directory or 'vinyl_crackle' in directory else 8000

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_stft_sample_rate, file) for file in files]

        for future in tqdm(as_completed(futures), total=len(files), desc=f"Checking {os.path.basename(directory)}"):
            file_path, sr = future.result()
            if sr != expected_sr:
                anomalies.append((file_path, sr))

    return anomalies


def main(directories):
    all_anomalies = {}
    for directory in directories:
        anomalies = process_directory(directory)
        if anomalies:
            all_anomalies[directory] = anomalies

    if all_anomalies:
        print("\nAnomalous files found:")
        for directory, anomalies in all_anomalies.items():
            print(f"\n{os.path.basename(directory)}:")
            for file_path, sr in anomalies:
                print(f"  {os.path.basename(file_path)}: {sr} Hz")
    else:
        print("\nNo anomalies found. All sample rates are as expected.")


if __name__ == "__main__":
    directories = [
        "../data/converted",
        "../data/low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ]
    main(directories)