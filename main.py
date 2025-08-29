from recorder.recorder_UI import RecorderUI
import tkinter as tk
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="#2C3E50")
    root.resizable(False, False)
    app = RecorderUI(root)

    window_width = 450
    window_height = 300
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = int((screen_width / 2) - (window_width / 2))
    y = int((screen_height / 2) - (window_height / 2))
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def on_closing():
        try:
            app.recorder.stop_recording()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
