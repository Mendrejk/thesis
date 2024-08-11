import os
import shutil
import multiprocessing
from pydub import AudioSegment
from tqdm import tqdm


def clean_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def process_file(file_info):
    input_path, output_dir = file_info
    filename = os.path.basename(input_path)
    file_id = os.path.splitext(filename)[0]

    # Load the WAV file
    audio = AudioSegment.from_wav(input_path)

    # Split into exactly 10-second segments
    segment_length = 10 * 1000  # 10 seconds in milliseconds
    num_full_segments = len(audio) // segment_length

    segments = []
    for i in range(num_full_segments):
        start = i * segment_length
        segment = audio[start:start + segment_length]
        output_filename = f"{file_id}-{i}.mp3"
        output_path = os.path.join(output_dir, output_filename)
        segment.export(output_path, format="mp3", bitrate="320k")
        segments.append(output_filename)

    # Check if there's a remaining segment and if it's exactly 10 seconds
    remaining_audio = audio[num_full_segments * segment_length:]
    if len(remaining_audio) == segment_length:
        output_filename = f"{file_id}-{num_full_segments}.mp3"
        output_path = os.path.join(output_dir, output_filename)
        remaining_audio.export(output_path, format="mp3", bitrate="320k")
        segments.append(output_filename)

    return filename, segments


def main():
    # Set the input and output directories
    input_dir = os.path.join("..", "data", "original")
    output_dir = os.path.join("..", "data", "converted", "mp3_segments")

    # Clean the output directory
    print("Cleaning output directory...")
    clean_directory(output_dir)

    # Get list of WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    file_infos = [(os.path.join(input_dir, f), output_dir) for f in wav_files]

    # Set up multiprocessing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Process files in parallel with progress bar
    with tqdm(total=len(file_infos), desc="Processing files") as pbar:
        for filename, segments in pool.imap_unordered(process_file, file_infos):
            pbar.update()
            print(f"Processed {filename}: created {len(segments)} 10-second segments")

    pool.close()
    pool.join()

    print("Processing complete!")


if __name__ == "__main__":
    main()
