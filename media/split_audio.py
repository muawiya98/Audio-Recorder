import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import STORAGE_PATH
from config import STORAGE_PATH
from pydub import AudioSegment
import pysrt
import json


def srt_to_ms(time_obj):
    """Convert pysrt time to milliseconds."""
    return (
        time_obj.hours * 3600 + time_obj.minutes * 60 + time_obj.seconds
    ) * 1000 + time_obj.milliseconds


def split_audio_by_srt(
    audio_path, srt_path, output_dir="chunks", json_path="chunks.json"
):
    audio = AudioSegment.from_file(audio_path)
    subs = pysrt.open(srt_path)
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    for i, sub in enumerate(subs, start=1):
        start_ms = srt_to_ms(sub.start)
        end_ms = srt_to_ms(sub.end)
        chunk = audio[start_ms:end_ms]
        chunk_filename = os.path.join(output_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_filename.replace(".wav", ".mp3"), format="mp3")
        key = sub.text.strip()
        metadata[i] = [key, chunk_filename]
        # print(f"Saved: {chunk_filename} ({sub.text})")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Audio splitting completed! Metadata saved to {json_path}")


if __name__ == "__main__":
    audio_path = os.path.join(STORAGE_PATH, '096.mp3')
    srt_surah = os.path.join(STORAGE_PATH, '096_words.srt')
    output_dir = os.path.join(STORAGE_PATH, 'chunk')
    json_path = os.path.join(STORAGE_PATH, 'chunks.json')
    split_audio_by_srt(
        audio_path=audio_path,
        srt_path=srt_surah,
        output_dir=output_dir,
        json_path=json_path,
    )
