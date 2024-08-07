import os
from pydub import AudioSegment

# Set the input and output directories
input_dir = os.path.join("..", "data", "original")
output_dir = os.path.join("..", "data", "converted", "mp3")

#print the current path
print(os.getcwd())

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set the bitrate for high-quality MP3 (320 kbps)
bitrate = "320k"

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        # Construct full file paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".mp3")

        # Load the WAV file
        audio = AudioSegment.from_wav(input_path)

        # Export as MP3 with high quality
        audio.export(output_path, format="mp3", bitrate=bitrate)

        print(f"Converted {filename} to MP3")

print("Conversion complete!")