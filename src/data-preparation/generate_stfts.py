import os
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import numpy as np
import librosa
import matplotlib.pyplot as plt
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def process_audio_file(file_path, output_dir, window_size=2048, hop_size=512):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)
        stft_scaled = signed_sqrt(stft.real) + 1j * signed_sqrt(stft.imag)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_stft.npz")
        np.savez_compressed(output_file, stft=stft_scaled, sr=sr, window_size=window_size, hop_size=hop_size)

        return file_path, stft_scaled
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return file_path, None


def process_directory(input_dir):
    mp3_dir = os.path.join(input_dir, "mp3")
    stft_dir = os.path.join(input_dir, "STFT")

    # Clear existing STFT files
    if os.path.exists(stft_dir):
        shutil.rmtree(stft_dir)
    os.makedirs(stft_dir, exist_ok=True)

    tasks = []
    for filename in os.listdir(mp3_dir):
        if filename.endswith(".mp3"):
            file_path = os.path.join(mp3_dir, filename)
            tasks.append((file_path, stft_dir))
    return tasks


def visualize_stft(file_path, stft):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                             y_axis='hz', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'STFT of {os.path.basename(file_path)}')
    plt.tight_layout()
    output_file = os.path.splitext(file_path)[0] + '_stft.png'
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved STFT visualization to {output_file}")


def main(directories):
    all_tasks = []
    for directory in directories:
        logging.info(f"Processing directory: {directory}")
        all_tasks.extend(process_directory(directory))

    num_cores = multiprocessing.cpu_count()
    logging.info(f"Using all {num_cores} CPU cores")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_audio_file, file_path, output_dir) for file_path, output_dir in all_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_path, stft = future.result()
            if stft is not None and "2353" in file_path:
                visualize_stft(file_path, stft)

    logging.info("STFT generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize STFTs for audio files.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/no_noise_ultra_low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    args = parser.parse_args()

    main(args.dirs)