import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_stft(file_path):
    with np.load(file_path) as data:
        return data['stft'], data['sr'], data['window_size'], data['hop_size']


def visualize_stft(ax, stft, sr, hop_size, title):
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                                   sr=sr, hop_length=hop_size, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title, fontsize=8)
    return img


def extract_track_id(filename):
    # Extract the first number found in the filename
    match = re.search(r'\d+', filename)
    return match.group() if match else None


def get_common_tracks(directories, num_samples=5):
    all_tracks = []
    for directory in directories:
        stft_dir = os.path.join(directory, 'STFT')
        if not os.path.exists(stft_dir):
            print(f"Warning: STFT directory not found: {stft_dir}")
            return []
        files = [f for f in os.listdir(stft_dir) if f.endswith('_stft.npz')]
        tracks = {}
        for file in files:
            track_id = extract_track_id(file)
            if track_id:
                tracks[track_id] = file
        all_tracks.append(tracks)

    common_track_ids = set.intersection(*[set(tracks.keys()) for tracks in all_tracks])
    if not common_track_ids:
        print("Error: No common tracks found across all directories.")
        print("Tracks in each directory:")
        for directory, tracks in zip(directories, all_tracks):
            print(f"  {directory}: {len(tracks)} tracks")
            for track_id in list(tracks.keys())[:5]:  # Print first 5 track IDs as examples
                print(f"    Track ID: {track_id}, File: {tracks[track_id]}")
            if len(tracks) > 5:
                print(f"    ... and {len(tracks) - 5} more")
        return []

    if len(common_track_ids) < num_samples:
        print(f"Warning: Only {len(common_track_ids)} common tracks found across all directories. Using all available.")
        num_samples = len(common_track_ids)

    selected_track_ids = random.sample(list(common_track_ids), num_samples)
    return selected_track_ids


def visualize_and_save_stfts(directories, output_dir, num_samples=5, max_workers=4):
    common_track_ids = get_common_tracks(directories, num_samples)
    if not common_track_ids:
        print("Visualization aborted due to lack of common tracks.")
        return

    all_stfts = []
    all_filenames = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for directory in directories:
            stft_dir = os.path.join(directory, 'STFT')
            files = [f for f in os.listdir(stft_dir) if f.endswith('_stft.npz')]
            for track_id in common_track_ids:
                matching_file = next((f for f in files if extract_track_id(f) == track_id), None)
                if matching_file:
                    futures.append(executor.submit(load_stft, os.path.join(stft_dir, matching_file)))
                    all_filenames.append(matching_file)

        for future in tqdm(futures, desc="Loading STFT files"):
            all_stfts.append(future.result())

    num_cols = len(common_track_ids)
    num_rows = len(directories)

    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    outer_grid = gridspec.GridSpec(num_rows, 1, hspace=0.4)

    for dir_idx, directory in enumerate(directories):
        dir_name = os.path.basename(os.path.normpath(directory))
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, num_cols,
                                                      subplot_spec=outer_grid[dir_idx], hspace=0.3, wspace=0.1)

        fig.add_subplot(outer_grid[dir_idx]).set_title(f"Directory: {dir_name}", fontsize=14, fontweight='bold')
        plt.axis('off')

        for i, track_id in enumerate(common_track_ids):
            ax = fig.add_subplot(inner_grid[0, i])
            file_idx = dir_idx * num_cols + i
            stft, sr, _, hop_size = all_stfts[file_idx]
            img = visualize_stft(ax, stft, sr, hop_size, f"Track {track_id}")

    fig.colorbar(img, ax=fig.axes, format='%+2.0f dB', aspect=30, pad=0.01)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'stft_visualization_{random.randint(1000, 9999)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Visualization saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize STFTs from specified directories and save as image.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/no_noise_ultra_low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    parser.add_argument("--output", default="../data/stft_images", help="Output directory for saved images")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to visualize from each directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for file loading")
    args = parser.parse_args()

    visualize_and_save_stfts(args.dirs, args.output, args.samples, args.workers)