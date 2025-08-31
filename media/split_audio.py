import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import STORAGE_PATH
import wave


def split_wav(input_file, chunk_length_sec, output_folder="chunks"):
    with wave.open(input_file, "rb") as wav:
        frame_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        samp_width = wav.getsampwidth()
        n_frames = wav.getnframes()

        duration = n_frames / float(frame_rate)
        print(f"Audio duration: {duration:.2f} seconds")

        frames_per_chunk = int(frame_rate * chunk_length_sec)

        os.makedirs(output_folder, exist_ok=True)

        chunk_index = 0
        while True:
            frames = wav.readframes(frames_per_chunk)
            if not frames:
                break

            chunk_path = os.path.join(output_folder, f"chunk_{chunk_index}.wav")
            with wave.open(chunk_path, "wb") as chunk:
                chunk.setnchannels(n_channels)
                chunk.setsampwidth(samp_width)
                chunk.setframerate(frame_rate)
                chunk.writeframes(frames)

            print(f"Saved {chunk_path}")
            chunk_index += 1


split_wav(os.path.join(STORAGE_PATH, "AlAalaq.wav"), 5)
