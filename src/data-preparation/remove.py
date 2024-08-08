import os
import datetime
import argparse


def remove_recent_npz_files(directories, cutoff_time):
    total_removed = 0
    for directory in directories:
        stft_dir = os.path.join(directory, "STFT")
        if not os.path.exists(stft_dir):
            print(f"STFT directory not found: {stft_dir}")
            continue

        for filename in os.listdir(stft_dir):
            if filename.endswith(".npz"):
                file_path = os.path.join(stft_dir, filename)
                file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_mtime > cutoff_time:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                    total_removed += 1

    print(f"Total files removed: {total_removed}")


def main():
    parser = argparse.ArgumentParser(description="Remove NPZ files created after a specific time.")
    parser.add_argument("--dirs", nargs="+", default=[
        "../data/converted",
        "../data/low_quality",
        "../data/no_noise_ultra_low_quality",
        "../data/ultra_low_quality",
        "../data/vinyl_crackle"
    ], help="List of directories to process")
    parser.add_argument("--cutoff", type=str, default="08:00",
                        help="Cutoff time in HH:MM format (24-hour clock)")
    args = parser.parse_args()

    today = datetime.date.today()
    cutoff_time = datetime.datetime.combine(today, datetime.datetime.strptime(args.cutoff, "%H:%M").time())

    print(f"Removing NPZ files created after {cutoff_time}")
    remove_recent_npz_files(args.dirs, cutoff_time)


if __name__ == "__main__":
    main()