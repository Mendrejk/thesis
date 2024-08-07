import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def signed_sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def process_audio_file(file_path, output_dir):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute STFT
    window_size = 2048
    hop_size = 512
    stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)

    # Apply non-linear scaling (signed square root)
    stft_scaled = signed_sqrt(stft.real) + 1j * signed_sqrt(stft.imag)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_stft.npy")

    # Save STFT
    np.save(output_file, stft_scaled)

    return file_path, stft_scaled


def process_directory(input_dir):
    mp3_dir = os.path.join(input_dir, "mp3")
    stft_dir = os.path.join(input_dir, "STFT")

    # Create STFT directory if it doesn't exist
    os.makedirs(stft_dir, exist_ok=True)

    # Process each MP3 file
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
    plt.show(block=False)


# List of directories to process
directories = [
    "../data/converted",
    "../data/low_quality",
    "../data/no_noise_ultra_low_quality",
    "../data/ultra_low_quality",
    "../data/vinyl_crackle"
]

# Gather all tasks
all_tasks = []
for directory in directories:
    print(f"Processing directory: {directory}")
    all_tasks.extend(process_directory(directory))

# Process files in parallel
num_cores = multiprocessing.cpu_count()
with ProcessPoolExecutor(max_workers=num_cores) as executor:
    futures = [executor.submit(process_audio_file, file_path, output_dir) for file_path, output_dir in all_tasks]

    for future in as_completed(futures):
        file_path, stft = future.result()
        if "2353" in file_path:
            visualize_stft(file_path, stft)

print("STFT generation complete!")
plt.show()  # This will block until all plot windows are closed