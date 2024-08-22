import os
import numpy as np
import matplotlib.pyplot as plt
import librosa


def load_stft_file(file_path):
    with np.load(file_path) as data:
        return data['stft']


def visualize_and_save_stft(clean_stft, vinyl_stft, sample_id, sr=22050, hop_length=512):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot clean STFT
    img1 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max),
                                    sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax1)
    ax1.set_title('Clean STFT', fontsize=16)
    fig.colorbar(img1, ax=ax1, format='%+2.0f dB')

    # Plot vinyl STFT
    img2 = librosa.display.specshow(librosa.amplitude_to_db(np.abs(vinyl_stft), ref=np.max),
                                    sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title('Vinyl Crackle STFT', fontsize=16)
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')

    plt.suptitle(f'STFT Comparison (Sample ID: {sample_id})', fontsize=20)
    plt.tight_layout()

    output_file = f'stft_comparison_{sample_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_file}")


def main():
    # Define the specific file to visualize
    sample_id = "2318-54"
    clean_dir = "../data/converted/stft_segments"
    vinyl_dir = "../data/vinyl_crackle/stft_segments"

    # Load clean and vinyl STFT files
    clean_file = os.path.join(clean_dir, f"{sample_id}_stft.npz")
    vinyl_file = os.path.join(vinyl_dir, f"{sample_id}_stft.npz")

    clean_stft = load_stft_file(clean_file)
    vinyl_stft = load_stft_file(vinyl_file)

    # Visualize and save STFTs
    visualize_and_save_stft(clean_stft, vinyl_stft, sample_id)


if __name__ == "__main__":
    main()