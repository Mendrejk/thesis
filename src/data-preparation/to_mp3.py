import os
import multiprocessing
from pydub import AudioSegment
from tqdm import tqdm

def process_file(file_info):
    input_path, output_dir = file_info
    filename = os.path.basename(input_path)
    file_id = os.path.splitext(filename)[0]

    # Load the WAV file
    audio = AudioSegment.from_wav(input_path)

    # Split into 10-second segments with 1-second overlap
    segment_length = 10 * 1000  # 10 seconds in milliseconds
    overlap = 1 * 1000  # 1 second overlap in milliseconds

    for i, start in enumerate(range(0, len(audio), segment_length - overlap)):
        segment = audio[start:start + segment_length]
        output_filename = f"{file_id}-{i}.mp3"
        output_path = os.path.join(output_dir, output_filename)
        segment.export(output_path, format="mp3", bitrate="320k")

    return filename

def main():
    # Set the input and output directories
    input_dir = os.path.join("..", "data", "original")
    output_dir = os.path.join("..", "data", "converted", "mp3_segments")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    file_infos = [(os.path.join(input_dir, f), output_dir) for f in wav_files]

    # Set up multiprocessing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Process files in parallel with progress bar
    with tqdm(total=len(file_infos), desc="Processing files") as pbar:
        for _ in pool.imap_unordered(process_file, file_infos):
            pbar.update()

    pool.close()
    pool.join()

    print("Processing complete!")

if __name__ == "__main__":
    main()