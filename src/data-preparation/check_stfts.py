import os
import numpy as np
import argparse
from tqdm import tqdm


def check_stft_file(file_path):
    try:
        with np.load(file_path) as data:
            # Check if required keys are present
            required_keys = ['stft', 'sr', 'window_size', 'hop_size']
            for key in required_keys:
                if key not in data:
                    return False, f"Missing key: {key}"

            # Check if STFT data is valid
            if data['stft'].size == 0:
                return False, "STFT data is empty"

            return True, "File is valid"
    except Exception as e:
        return False, str(e)


def scan_directory(directory):
    stft_dir = os.path.join(directory, 'STFT')
    if not os.path.exists(stft_dir):
        print(f"Warning: STFT directory not found in {directory}")
        return []

    return [os.path.join(stft_dir, f) for f in os.listdir(stft_dir) if f.endswith('_stft.npz')]


def check_all_stfts(directories):
    all_stfts = []
    for directory in directories:
        all_stfts.extend(scan_directory(directory))

    if not all_stfts:
        print("No STFT files found in the specified directories.")
        return

    invalid_files = []

    for stft_file in tqdm(all_stfts, desc="Checking STFT files"):
        is_valid, message = check_stft_file(stft_file)
        if not is_valid:
            invalid_files.append((stft_file, message))

    # Report results
    print(f"\nTotal STFT files checked: {len(all_stfts)}")
    print(f"Valid files: {len(all_stfts) - len(invalid_files)}")
    print(f"Invalid files: {len(invalid_files)}")

    if invalid_files:
        print("\nList of invalid files:")
        for file, error in invalid_files:
            print(f"{file}: {error}")

    return invalid_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check validity of STFT files in specified directories.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/ultra_low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    args = parser.parse_args()

    invalid_files = check_all_stfts(args.dirs)

    # Optionally, you can add code here to delete or move invalid files
    if invalid_files:
        user_input = input("Do you want to delete invalid files? (y/n): ")
        if user_input.lower() == 'y':
            for file, _ in invalid_files:
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")