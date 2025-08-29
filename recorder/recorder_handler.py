from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from config import STORAGE_PATH
import sounddevice as sd
import numpy as np
import threading
import wave
import os


class RecorderHandler:

    # 🔹 Load ASR model once (class variable)
    MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
    MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
    PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
    MODEL.generation_config.no_timestamps_token_id = (
        PROCESSOR.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    )
    PIPE = pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        tokenizer=PROCESSOR.tokenizer,
        feature_extractor=PROCESSOR.feature_extractor,
    )

    def __init__(
        self,
        filename="record",
        extension=".wav",
        samplerate=16000,
        channels=1,
        chunk_size=1024,
    ):
        self.filename = f"{filename}{extension}"
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_size = chunk_size
        self.frames = []
        self.is_recording = False
        self.is_paused = False
        self.thread = None

    def _callback(self, indata, frames, time, status):
        if self.is_recording and not self.is_paused:
            self.frames.append(indata.copy())
            # Convert chunk → text asynchronously
            threading.Thread(
                target=self.transcribe_chunk, args=(indata.copy(),)
            ).start()

    def transcribe_chunk(self, chunk):
        """Convert chunk to text using Hugging Face pipeline"""
        # Convert float32 PCM → int16 PCM
        pcm16 = (chunk * 32767).astype(np.int16)
        # Save chunk as temporary WAV
        temp_path = self.RECORDINGS_DIR / "temp_chunk.wav"
        with wave.open(str(temp_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(pcm16.tobytes())

        # Run ASR
        result = self.PIPE(str(temp_path), return_timestamps=True)
        print("📝 Chunk text:", result["text"])

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.is_paused = False
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=self.chunk_size,
            callback=self._callback,
        ):
            while self.is_recording:
                sd.sleep(100)

    def pause_recording(self):
        if self.is_recording:
            self.is_paused = not self.is_paused

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.thread:
            self.thread.join()
        self._save_to_file()

    def _save_to_file(self):
        if not self.frames:
            return
        data = np.concatenate(self.frames, axis=0)
        data = (data * 32767).astype(np.int16)
        os.makedirs(STORAGE_PATH, exist_ok=True)
        with wave.open(os.path.join(STORAGE_PATH, self.filename), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.samplerate)
            wf.writeframes(data.tobytes())
