import os
import sys
import subprocess
import numpy as np
import librosa
import concurrent.futures


def extract_audio_from_video(video_path, output_audio_path):
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-q:a",
                "0",
                "-map",
                "a",
                output_audio_path,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to extract audio from {video_path}")
        return False


def find_audio_match(target_audio_path, source_audio_path):
    try:
        target_audio, sr_target = librosa.load(target_audio_path, sr=None)
        source_audio, sr_source = librosa.load(source_audio_path, sr=None)

        result = np.correlate(source_audio, target_audio, mode="valid")
        start_sample = np.argmax(result)
        duration_target = len(target_audio)
        end_sample = start_sample + duration_target

        if start_sample + duration_target > len(source_audio):
            return None, None

        start_time = start_sample / sr_source
        end_time = end_sample / sr_source

        return start_time, end_time
    except Exception as e:
        print(f"Error finding audio match: {e}")
        return None, None


def trim_video(input_video_path, start_time, end_time, output_video_path):
    start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}.{int((start_time % 1) * 1000):03}"
    end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int((end_time % 60)):02}.{int((end_time % 1) * 1000):03}"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_video_path,
                "-vf",
                f"select='not(between(t,{start_time},{end_time}))',setpts=N/FRAME_RATE/TB",
                "-af",
                f"aselect='not(between(t,{start_time},{end_time}))',asetpts=N/SR/TB",
                output_video_path,
            ],
            check=True,
        )
        print(f"Trimmed video saved to {output_video_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to trim video {input_video_path}")
        return False


def cleanup(files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def process_video(input_video_path, target_audio_path):
    extracted_audio_path = os.path.splitext(input_video_path)[0] + "_audio.mp3"
    output_video_path = (
        os.path.splitext(input_video_path)[0]
        + "_trimmed"
        + os.path.splitext(input_video_path)[1]
    )

    if not extract_audio_from_video(input_video_path, extracted_audio_path):
        return

    start_time, end_time = find_audio_match(target_audio_path, extracted_audio_path)
    if start_time is None or end_time is None:
        print(f"No valid match found for {input_video_path}. Skipping.")
        cleanup([extracted_audio_path])
        return

    if trim_video(input_video_path, start_time, end_time, output_video_path):
        cleanup([extracted_audio_path])


def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file_or_directory> <target_audio_file>")
        return

    input_path = sys.argv[1]
    target_audio_path = "./outro.mp3"

    if os.path.isfile(input_path) and input_path.endswith(".mp4"):
        process_video(input_path, target_audio_path)
    elif os.path.isdir(input_path):
        mp4_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".mp4")
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_video, mp4_file, target_audio_path)
                for mp4_file in mp4_files
            ]
            concurrent.futures.wait(futures)
    else:
        print(
            "Invalid input path. Must be a .mp4 file or a directory containing .mp4 files."
        )


if __name__ == "__main__":
    main()
