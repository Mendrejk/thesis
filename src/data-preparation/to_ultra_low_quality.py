import os
import shutil
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set the input and output directories
input_dir = os.path.join("..", "data", "converted", "mp3_segments")
output_dir = os.path.join("..", "data", "ultra_low_quality", "mp3_segments")

# Set the parameters for ultra-low-quality conversion
bitrate = "8k"
sample_rate = 4000


def clear_directory(directory):
    """Clear all files in the specified directory."""
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)


def process_file(filename):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Load the MP3 file
    audio = AudioSegment.from_mp3(input_path)

    # Reduce the quality (keeping stereo)
    ultra_low_quality_audio = audio.set_frame_rate(sample_rate)

    # Export as ultra-low-quality MP3
    ultra_low_quality_audio.export(
        output_path,
        format="mp3",
        bitrate=bitrate,
        parameters=["-q:a", "9"]
    )

    return filename


def main():
    # Clear the output directory
    clear_directory(output_dir)
    print(f"Cleared output directory: {output_dir}")

    # Get all MP3 files in the input directory
    mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".mp3")]

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = [executor.submit(process_file, filename) for filename in mp3_files]

        # Create a progress bar
        with tqdm(total=len(mp3_files), desc="Converting files") as pbar:
            for future in as_completed(futures):
                filename = future.result()
                pbar.update(1)
                pbar.set_postfix_str(f"Processed {filename}")

    print("Ultra-low-quality conversion complete!")


if __name__ == "__main__":
    main()
