import os
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import warnings


def list_runs(log_dir):
    runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith('run_')]
    return sorted(runs, reverse=True)


def list_epochs(run_dir):
    epochs = [d for d in os.listdir(run_dir) if d.startswith('epoch_') and os.path.isdir(os.path.join(run_dir, d))]
    return sorted(epochs, key=lambda x: int(x.split('_')[1]))


def list_samples(epoch_dir):
    samples = [f for f in os.listdir(epoch_dir) if f.endswith('.pt')]
    return sorted(samples, key=lambda x: int(x.split('_')[1].split('.')[0]))


def convert_spectrogram_to_audio(spectrogram):
    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"Spectrogram dtype: {spectrogram.dtype}")

    # Assuming the spectrogram shape is [12, 2, 1025, 862]
    # We'll process each of the 12 segments separately
    n_fft = 2048
    hop_length = 512
    win_length = 2048

    waveforms = []
    for i in range(spectrogram.shape[0]):
        spec_segment = spectrogram[i]  # Shape: [2, 1025, 862]

        # Convert to complex
        complex_spec = torch.complex(spec_segment[0], spec_segment[1])

        segment_waveform = torch.istft(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=spectrogram.device),
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False
        )
        waveforms.append(segment_waveform)

    # Concatenate all segments
    full_waveform = torch.cat(waveforms, dim=0)
    return full_waveform


def save_as_mp3(waveform, sample_rate, output_path):
    # Convert to numpy array
    audio_numpy = waveform.cpu().numpy()

    # Normalize audio to 16-bit range
    audio_numpy = np.int16(audio_numpy / np.max(np.abs(audio_numpy)) * 32767)

    # Save as WAV first
    temp_wav_path = output_path.replace('.mp3', '_temp.wav')
    wavfile.write(temp_wav_path, sample_rate, audio_numpy)

    # Convert WAV to MP3
    audio = AudioSegment.from_wav(temp_wav_path)
    audio.export(output_path, format="mp3")

    # Remove temporary WAV file
    os.remove(temp_wav_path)


def main():
    log_dir = './logs'  # Update this to your log directory path

    # List available runs
    runs = list_runs(log_dir)
    print("Available runs:")
    for i, run in enumerate(runs):
        print(f"{i + 1}. {run}")

    # Select run
    run_index = int(input("Enter the number of the run you want to use: ")) - 1
    selected_run = runs[run_index]
    run_dir = os.path.join(log_dir, selected_run)

    # List epochs in the selected run
    epochs = list_epochs(run_dir)
    print("\nAvailable epochs:")
    for i, epoch in enumerate(epochs):
        print(f"{i + 1}. {epoch}")

    # Select epochs
    epoch_indices = input("Enter the numbers of the epochs you want to convert (comma-separated): ")
    selected_epochs = [epochs[int(i) - 1] for i in epoch_indices.split(',')]

    # Process selected epochs
    for epoch in selected_epochs:
        epoch_dir = os.path.join(run_dir, epoch)
        samples = list_samples(epoch_dir)

        print(f"\nProcessing {epoch}:")
        for sample in samples:
            sample_path = os.path.join(epoch_dir, sample)
            output_path = sample_path.replace('.pt', '.mp3')

            # Load the tensor with a warning filter
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectrogram = torch.load(sample_path, map_location='cpu')

            # Convert spectrogram to audio
            waveform = convert_spectrogram_to_audio(spectrogram)

            # Save as MP3
            save_as_mp3(waveform, sample_rate=44100, output_path=output_path)

            print(f"Converted {sample} to MP3")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()