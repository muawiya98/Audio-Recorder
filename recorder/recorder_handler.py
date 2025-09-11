import enum
from config import STORAGE_PATH, get_data
from fetch_audio import fetch_audio
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
        self.run_stt = True
        self.transcribe = None
        self.time_counter = 0
        self.time_stamp = time.time()
        self.words, self.paths = get_data()
        self.last_word_index = 0

    def _callback(self, indata, frames, time, status):
        if self.is_recording and not self.is_paused:
            self.frames.append(indata.copy())
            self.buffer.append(indata.copy())
            if len(self.buffer) * self.chunk_size >= self.samplerate * 5:  # ~5 seconds
                data = np.concatenate(self.buffer, axis=0)
                self.buffer = []
                threading.Thread(target=self.transcribe_chunk, args=(data,)).start()

    def fetch_audio_path(self, transcribe):
        words = transcribe.split()
        old_index = self.last_word_index
        for i in range(0, len(words)):
            for ind in range(self.last_word_index, len(self.words)):
                if words[i] in self.words[ind]:
                    old_index = ind
                    break
        self.last_word_index = old_index
        # if old_index == self.last_word_index or old_index < (
        #     self.last_word_index + (len(words) // 2)
        # ):
        #     return self.paths[old_index]
        # return None

    def transcribe_chunk(self, chunk, silence_threshold=50):
        pcm16 = (chunk * 32767).astype(np.int16)
        if self.time_counter > 12:
            word_path = self.paths[self.last_word_index + 1]
            self.time_counter = 0
            playsound(word_path)
            time.sleep(1)
            self.time_stamp = time.time()

        rms = np.sqrt(np.mean(pcm16.astype(np.float32) ** 2))
        if rms < silence_threshold:  # skip if too quiet
            self.time_counter = time.time() - self.time_stamp
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            with wave.open(tmpfile.name, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.samplerate)
                wf.writeframes(pcm16.tobytes())
            tmp_path = tmpfile.name

        if self.run_stt:
            with open(tmp_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe", file=audio_file, language="ar"
                )
                self.transcribe = response.text
                print("Chunk text:", response.text)

        if self.transcribe:
            # output_file =
            self.fetch_audio_path(self.transcribe)
            self.transcribe = None
            # if output_file:
            #     self.run_stt = False
            # playsound(output_file)
            # time.sleep(1)
            # self.run_stt = True

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
