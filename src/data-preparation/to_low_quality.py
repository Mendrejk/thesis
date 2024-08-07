import os
from pydub import AudioSegment

# Set the input and output directories
input_dir = os.path.join("..", "data", "converted", "mp3")
output_dir = os.path.join("..", "data", "low_quality", "mp3")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set the parameters for low-quality conversion
bitrate = "32k"  # Very low bitrate
sample_rate = 8000  # Low sample rate (8 kHz)

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):
        # Construct full file paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"low_quality_{filename}")

        # Load the MP3 file
        audio = AudioSegment.from_mp3(input_path)

        # Reduce the quality
        low_quality_audio = audio.set_frame_rate(sample_rate).set_channels(1)

        # Export as low-quality MP3
        low_quality_audio.export(output_path, format="mp3", bitrate=bitrate)

        print(f"Created low-quality version of {filename}")

print("Low-quality conversion complete!")