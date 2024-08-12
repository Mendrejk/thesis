import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import librosa
import argparse
import psutil
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the maximum RAM usage (in bytes)
MAX_RAM = 22 * 1024 * 1024 * 1024  # 22GB
MAX_CORES = 12


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def clear_directory(directory):
    """Clear all files in the specified directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)


def get_target_shape(directories, window_size=2048, hop_size=512):
    max_sr = 0
    max_duration = 0
    for directory in directories:
        mp3_segments_dir = os.path.join(directory, "mp3_segments")
        for filename in os.listdir(mp3_segments_dir):
            if filename.endswith(".mp3"):
                file_path = os.path.join(mp3_segments_dir, filename)
                audio, sr = librosa.load(file_path, sr=None, duration=1)  # Load just 1 second to get sr
                max_sr = max(max_sr, sr)
                duration = librosa.get_duration(filename=file_path)
                max_duration = max(max_duration, duration)

    num_freq_bins = window_size // 2 + 1
    num_time_frames = int(np.ceil(max_duration * max_sr / hop_size))
    return (num_freq_bins, num_time_frames)


def process_audio_file(file_path, output_dir, window_size=2048, hop_size=512):
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_stft.npz")

        # Skip if the file already exists
        if os.path.exists(output_file):
            return False, (0, 0)  # Skipped, return dummy shape

        audio, sr = librosa.load(file_path, sr=None)
        stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)
        stft_scaled = signed_sqrt(stft.real) + 1j * signed_sqrt(stft.imag)

        np.savez_compressed(output_file, stft=stft_scaled, sr=sr, window_size=window_size, hop_size=hop_size)

        return True, stft_scaled.shape  # Processed, return actual shape
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return False, (0, 0)  # Error, counted as skipped, return dummy shape


def process_directory(input_dir, should_clear):
    mp3_segments_dir = os.path.join(input_dir, "mp3_segments")
    stft_segments_dir = os.path.join(input_dir, "stft_segments")

    # Clear and recreate STFT segments directory if requested
    if should_clear:
        clear_directory(stft_segments_dir)
        logging.info(f"Cleared output directory: {stft_segments_dir}")
    else:
        os.makedirs(stft_segments_dir, exist_ok=True)

    tasks = []
    for filename in os.listdir(mp3_segments_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_segments_dir, filename)
            tasks.append((file_path, stft_segments_dir))
    return tasks


def main(directories):
    should_clear = input("Do you want to clear existing STFT files? (y/N): ").lower() == 'y'

    all_tasks = []
    for directory in directories:
        logging.info(f"Processing directory: {directory}")
        all_tasks.extend(process_directory(directory, should_clear))

    num_cores = MAX_CORES
    logging.info(f"Using {num_cores} CPU cores and up to {MAX_RAM / (1024 ** 3):.1f} GB of RAM")

    processed_count = 0
    skipped_count = 0
    max_shape = (0, 0)

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_audio_file, file_path, output_dir)
                   for file_path, output_dir in all_tasks]

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result, shape = future.result()
                if result:
                    processed_count += 1
                    max_shape = (max(max_shape[0], shape[0]), max(max_shape[1], shape[1]))
                else:
                    skipped_count += 1

                pbar.set_description(f"Processed: {processed_count}, Skipped: {skipped_count}")
                pbar.update(1)

                # Check available memory and wait if necessary
                while psutil.virtual_memory().available < MAX_RAM * 0.1:
                    pbar.set_description("Waiting for memory to be freed...")
                    psutil.wait_procs(psutil.Process().children(), timeout=5)

    logging.info(f"STFT generation complete! Processed: {processed_count}, Skipped: {skipped_count}")
    logging.info(f"Maximum STFT shape: {max_shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate STFTs for audio files with resource constraints.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    args = parser.parse_args()

    main(args.dirs)