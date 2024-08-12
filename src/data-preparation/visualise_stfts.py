import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def load_and_merge_stfts(directory, track_number):
    stft_dir = os.path.join(directory, 'stft_segments')
    files = sorted([f for f in os.listdir(stft_dir) if f.startswith(f"{track_number}-") and f.endswith('_stft.npz')])

    merged_stft = None
    sr = None
    window_size = None
    hop_size = None

    for file in files:
        with np.load(os.path.join(stft_dir, file)) as data:
            stft = data['stft']
            if merged_stft is None:
                merged_stft = stft
                sr = data['sr']
                window_size = data['window_size']
                hop_size = data['hop_size']
            else:
                merged_stft = np.hstack((merged_stft, stft))

    return merged_stft, sr, window_size, hop_size


def visualize_stft(ax, stft, sr, hop_size, title):
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                                   sr=sr, hop_length=hop_size, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    return img


def get_common_tracks(directories):
    all_tracks = []
    for directory in directories:
        stft_dir = os.path.join(directory, 'stft_segments')
        if not os.path.exists(stft_dir):
            print(f"Warning: STFT directory not found: {stft_dir}")
            return []
        files = [f for f in os.listdir(stft_dir) if f.endswith('_stft.npz')]
        tracks = set(f.split('-')[0] for f in files)
        all_tracks.append(tracks)

    return list(set.intersection(*all_tracks))


def visualize_and_save_stfts(directories, output_dir, num_samples=5, max_workers=4):
    common_tracks = get_common_tracks(directories)
    if not common_tracks:
        print("Visualization aborted due to lack of common tracks.")
        return

    if len(common_tracks) < num_samples:
        print(f"Warning: Only {len(common_tracks)} common tracks found across all directories. Using all available.")
        num_samples = len(common_tracks)

    selected_tracks = random.sample(common_tracks, num_samples)

    all_stfts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for directory in directories:
            for track in selected_tracks:
                futures.append(executor.submit(load_and_merge_stfts, directory, track))

        for future in tqdm(futures, desc="Loading and merging STFT files"):
            all_stfts.append(future.result())

    num_cols = len(selected_tracks)
    num_rows = len(directories)

    fig = plt.figure(figsize=(6 * num_cols, 4 * num_rows + 1))
    outer_grid = gridspec.GridSpec(num_rows + 1, 1, height_ratios=[0.5] + [1] * num_rows, hspace=0.3)

    # Add a title for the entire figure
    fig.add_subplot(outer_grid[0]).set_title("Full Track STFT Visualization Across Different Audio Qualities",
                                             fontsize=16, fontweight='bold')
    plt.axis('off')

    for dir_idx, directory in enumerate(directories):
        dir_name = os.path.basename(os.path.normpath(directory))
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, num_cols,
                                                      subplot_spec=outer_grid[dir_idx + 1], hspace=0.3, wspace=0.3)

        for i, track in enumerate(selected_tracks):
            ax = fig.add_subplot(inner_grid[0, i])
            file_idx = dir_idx * num_cols + i
            stft, sr, _, hop_size = all_stfts[file_idx]
            img = visualize_stft(ax, stft, sr, hop_size, f"{dir_name}\nTrack {track}")

            if dir_idx == num_rows - 1:  # Only add x-label for the bottom row
                ax.set_xlabel('Time')
            if i == 0:  # Only add y-label for the leftmost column
                ax.set_ylabel('Frequency (Hz)')

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')
    cbar_ax.set_ylabel('Magnitude (dB)', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.91, 1])  # Adjust layout to make room for colorbar

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'full_track_stft_visualization_{random.randint(1000, 9999)}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Visualization saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize full track STFTs from specified directories and save as image.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    parser.add_argument("--output", default="../data/stft_images", help="Output directory for saved images")
    parser.add_argument("--samples", type=int, default=5, help="Number of tracks to visualize from each directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads for file loading")
    args = parser.parse_args()

    visualize_and_save_stfts(args.dirs, args.output, args.samples, args.workers)