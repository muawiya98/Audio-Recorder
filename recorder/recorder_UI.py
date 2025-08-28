from recorder.recorder_handler import RecorderHandler
from tkinter import ttk
import tkinter as tk


class RecorderUI:


    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Audio Recorder")
        self.recorder = RecorderHandler()

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton", font=("Segoe UI", 12, "bold"), padding=10, foreground="white"
        )
        style.configure("Start.TButton", background="#4CAF50")
        style.map("Start.TButton", background=[("active", "#45a049")])
        style.configure("Pause.TButton", background="#FF9800")
        style.map("Pause.TButton", background=[("active", "#e68900")])
        style.configure("Stop.TButton", background="#F44336")
        style.map("Stop.TButton", background=[("active", "#d32f2f")])

        self.frame = ttk.Frame(root, padding=20)
        self.frame.pack(expand=True, fill="both")

        self.start_btn = ttk.Button(
            self.frame,
            text="Start",
            style="Start.TButton",
            command=self.start_recording,
        )
        self.start_btn.pack(pady=10, fill="x")

        self.pause_btn = ttk.Button(
            self.frame,
            text="Pause/Resume",
            style="Pause.TButton",
            command=self.pause_recording,
        )
        self.pause_btn.pack(pady=10, fill="x")

        self.stop_btn = ttk.Button(
            self.frame, text="Stop", style="Stop.TButton", command=self.stop_recording
        )
        self.stop_btn.pack(pady=10, fill="x")

        self.status_label = tk.Label(
            self.frame, text="⏹️ Idle", fg="white", bg="#2C3E50", font=("Segoe UI", 11)
        )
        self.status_label.pack(pady=10)

        self.progress = ttk.Progressbar(
            self.frame, orient="horizontal", length=250, mode="indeterminate"
        )
        self.progress.pack(pady=5)

    def start_recording(self):
        self.recorder.start_recording()
        self.status_label.config(text="🎙️ Recording...", fg="#4CAF50")
        self.progress.start(50)

    def pause_recording(self):
        self.recorder.pause_recording()
        if self.recorder.is_paused:
            self.status_label.config(text="⏸️ Paused", fg="#FF9800")
            self.progress.stop()
        else:
            self.status_label.config(text="🎙️ Recording...", fg="#4CAF50")
            self.progress.start(50)

    def stop_recording(self):
        self.recorder.stop_recording()
        self.status_label.config(text="✅ Stopped (saved)", fg="#F44336")
        self.progress.stop()
