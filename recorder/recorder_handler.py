from config import STORAGE_PATH
import sounddevice as sd
import numpy as np
import threading
import wave
import os


class RecorderHandler:

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
