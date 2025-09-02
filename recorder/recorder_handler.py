from fetch_audio import fetch_audio
from config import STORAGE_PATH
from playsound import playsound
from openai import OpenAI
import sounddevice as sd
import numpy as np
import threading
import tempfile
import time
import wave
import os


from dotenv import load_dotenv
load_dotenv()


class RecorderHandler:
    api_key = os.getenv("OPENAI_API_KEY")

    # Init OpenAI client
    client = OpenAI(api_key=api_key)

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
        self.buffer = []

    def _callback(self, indata, frames, time, status):
        if self.is_recording and not self.is_paused:
            self.frames.append(indata.copy())
            self.buffer.append(indata.copy())
            if len(self.buffer) * self.chunk_size >= self.samplerate * 3:  # ~3 seconds
                data = np.concatenate(self.buffer, axis=0)
                self.buffer = []
                threading.Thread(target=self.transcribe_chunk, args=(data,)).start()

    def transcribe_chunk(self, chunk):
        pcm16 = (chunk * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            with wave.open(tmpfile.name, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(pcm16.tobytes())
            tmp_path = tmpfile.name

        # Transcribe with OpenAI Whisper
        with open(tmp_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=audio_file, language="ar"
            )
        print("Chunk text:", response.text)
        output_file, duration = fetch_audio(response.text)
        if output_file:
            # self.pause_recording()
            playsound(output_file)
            time.sleep((duration/1000)+0.1)
            os.remove(output_file)
            # self.start_recording()
        else:
            print("Sentence not found in SRT")
            

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
