import os
from pydub import AudioSegment
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

def add_noise(audio_segment, noise_level=0.01):
    samples = np.array(audio_segment.get_array_of_samples())
    noise = np.random.randn(len(samples)) * noise_level * np.max(np.abs(samples))
    noisy_samples = (samples + noise).astype(samples.dtype)
    return AudioSegment(noisy_samples.tobytes(),
                        frame_rate=audio_segment.frame_rate,
                        sample_width=audio_segment.sample_width,
                        channels=audio_segment.channels)

def process_file(filename, input_dir, output_dir, bitrate, sample_rate):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"ultra_low_quality_{filename}")

    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_path)

    # Reduce the quality
    ultra_low_quality_audio = (
        audio
        .set_frame_rate(sample_rate)
        # .set_channels(1)
        # .compress_dynamic_range()
    )

    # Add noise
    # ultra_low_quality_audio = add_noise(ultra_low_quality_audio, noise_level=0.02)

    # Export as ultra-low-quality MP3
    ultra_low_quality_audio.export(
        output_path,
        format="mp3",
        bitrate=bitrate,
        parameters=["-q:a", "9"]
    )

    return f"Created ultra-low-quality version of {filename}"

def main():
    # Set the input and output directories
    input_dir = os.path.join("..", "data", "converted")
    output_dir = os.path.join("..", "data", "no_noise_ultra_low_quality")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the parameters for ultra-low-quality conversion
    bitrate = "8k"
    sample_rate = 4000

    # Get list of MP3 files
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    # Set up the process pool
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    # Create a partial function with fixed arguments
    process_file_partial = partial(process_file, input_dir=input_dir, output_dir=output_dir,
                                   bitrate=bitrate, sample_rate=sample_rate)

    # Process files in parallel
    results = pool.map(process_file_partial, mp3_files)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Print results
    for result in results:
        print(result)

    print("Ultra-low-quality conversion complete!")

if __name__ == "__main__":
    main()