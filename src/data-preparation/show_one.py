import os
import numpy as np
import matplotlib.pyplot as plt
import librosa


def load_first_stft(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            with np.load(file_path) as data:
                # Assuming the STFT data is stored under a key 'stft'
                # If it's stored under a different key, you'll need to change this
                stft = data['stft']
                return stft, filename
    return None, None


def visualize_stft(stft, output_file, sr=22050, hop_length=512):
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                             sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Magnitude')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def main():
    stft_dir = "../data/converted/stft_segments"
    output_dir = "./stft_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading the first STFT file...")
    stft_data, filename = load_first_stft(stft_dir)

    if stft_data is not None:
        print(f"STFT loaded successfully from file: {filename}")
        print(f"STFT shape: {stft_data.shape}")

        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_visualization.png")
        print(f"Saving the STFT visualization to {output_file}...")
        visualize_stft(stft_data, output_file)
        print("Visualization saved successfully.")
    else:
        print("No STFT files found in the specified directory.")


if __name__ == "__main__":
    main()