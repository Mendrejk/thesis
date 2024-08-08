import os
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from tqdm import tqdm
import numpy as np
import librosa
import argparse
import psutil
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the maximum RAM usage (in bytes)
MAX_RAM = 16 * 1024 * 1024 * 1024  # 16GB
MAX_CORES = 12


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def process_audio_file(file_path, output_dir, window_size=2048, hop_size=512):
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_stft.npz")

        # Skip if the file already exists
        if os.path.exists(output_file):
            logging.info(f"Skipping existing file: {output_file}")
            return file_path, None

        audio, sr = librosa.load(file_path, sr=None)
        stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)
        stft_scaled = signed_sqrt(stft.real) + 1j * signed_sqrt(stft.imag)

        np.savez_compressed(output_file, stft=stft_scaled, sr=sr, window_size=window_size, hop_size=hop_size)

        return file_path, output_file
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return file_path, None


def process_directory(input_dir):
    mp3_dir = os.path.join(input_dir, "mp3")
    stft_dir = os.path.join(input_dir, "STFT")

    # Create STFT directory if it doesn't exist
    os.makedirs(stft_dir, exist_ok=True)

    tasks = []
    for filename in os.listdir(mp3_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_dir, filename)
            tasks.append((file_path, stft_dir))
    return tasks


def main(directories):
    all_tasks = []
    for directory in directories:
        logging.info(f"Processing directory: {directory}")
        all_tasks.extend(process_directory(directory))

    num_cores = min(MAX_CORES, multiprocessing.cpu_count())
    logging.info(f"Using {num_cores} CPU cores")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for file_path, output_dir in all_tasks:
            # Check if we have enough memory to start a new task
            while psutil.virtual_memory().available < MAX_RAM * 0.2:  # Ensure at least 20% of MAX_RAM is available
                # Wait for some tasks to complete
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    file_path, output_file = future.result()
                    if output_file:
                        logging.info(f"Completed processing: {file_path}")

            futures.append(executor.submit(process_audio_file, file_path, output_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path, output_file = future.result()
            if output_file:
                logging.info(f"Completed processing: {file_path}")

    logging.info("STFT generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate STFTs for audio files with resource constraints.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/no_noise_ultra_low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    args = parser.parse_args()

    main(args.dirs)