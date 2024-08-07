import os
import numpy as np
from pydub import AudioSegment
from scipy import signal
from multiprocessing import Pool, cpu_count
from functools import partial


def generate_vinyl_crackle(duration_ms, sample_rate):
    num_samples = int(duration_ms * sample_rate / 1000)
    samples = np.zeros(num_samples)

    # Generate sparse pops and crackles
    event_density = 0.0001 # Doubled from 0.00005 to make events twice as common
    event_positions = np.random.randint(0, num_samples, int(num_samples * event_density))

    for pos in event_positions:
        event_type = np.random.choice(['pop', 'crackle', 'scratch'])

        if event_type == 'pop':
            # Short, sharp pop
            duration = np.random.randint(5, 15)
            event = np.random.exponential(0.01, duration)
            event = event * np.hanning(duration)
        elif event_type == 'crackle':
            # Series of tiny clicks
            duration = np.random.randint(20, 50)
            event = np.random.normal(0, 0.01, duration)
            event = event * (np.random.random(duration) > 0.7)  # Make it sparse
        else:  # scratch
            # Longer, more continuous sound
            duration = np.random.randint(50, 200)
            event = np.random.normal(0, 0.05, duration)
            event = event * np.hanning(duration)

        # Add event to samples
        end_pos = min(pos + len(event), num_samples)
        samples[pos:end_pos] += event[:end_pos - pos]

    # Apply a bandpass filter to focus on mid-range frequencies
    nyquist = sample_rate / 2
    low = 500 / nyquist
    high = 7000 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    samples = signal.lfilter(b, a, samples)

    # Normalize
    samples = samples / np.max(np.abs(samples))

    return samples

def add_vinyl_crackle(audio, crackle_level=0.5):
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    vinyl_crackle = generate_vinyl_crackle(len(audio), sample_rate)

    # Ensure vinyl_crackle has the same length as samples
    if len(vinyl_crackle) > len(samples):
        vinyl_crackle = vinyl_crackle[:len(samples)]
    elif len(vinyl_crackle) < len(samples):
        vinyl_crackle = np.pad(vinyl_crackle, (0, len(samples) - len(vinyl_crackle)))

    # Increase the crackle level
    vinyl_crackle = (vinyl_crackle * crackle_level * np.max(np.abs(samples)) * 2).astype(samples.dtype)

    noisy_samples = samples + vinyl_crackle
    noisy_samples = np.clip(noisy_samples, np.iinfo(samples.dtype).min, np.iinfo(samples.dtype).max)

    return AudioSegment(noisy_samples.tobytes(),
                        frame_rate=audio.frame_rate,
                        sample_width=audio.sample_width,
                        channels=audio.channels)

def process_file(filename, input_dir, output_dir, crackle_level):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"vinyl_crackle_{filename}")

    # Load the audio file
    audio = AudioSegment.from_file(input_path, format="mp3")

    # Add vinyl crackle
    noisy_audio = add_vinyl_crackle(audio, crackle_level)

    # Export as MP3
    noisy_audio.export(output_path, format="mp3")

    return f"Added vinyl crackle to {filename}"

def main():
    # Set the input and output directories
    input_dir = os.path.join("..", "data", "converted", "mp3")
    output_dir = os.path.join("..", "data", "vinyl_crackle", "mp3")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the crackle level
    crackle_level = 0.5  # Adjust as needed

    # Get list of MP3 files
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    # Set up the process pool
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU core free
    pool = Pool(processes=num_processes)

    # Create a partial function with fixed arguments
    process_file_partial = partial(process_file, input_dir=input_dir, output_dir=output_dir,
                                   crackle_level=crackle_level)

    # Process files in parallel
    results = pool.map(process_file_partial, mp3_files)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    # Print results
    for result in results:
        print(result)

    print("Vinyl crackle addition complete!")

if __name__ == "__main__":
    main()