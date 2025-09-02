# Audio Recorder

A lightweight, user-friendly audio recording application built with Python and Tkinter. This application provides a simple interface for recording, pausing, and saving audio files with professional-quality output.

## Features

- **Simple & Intuitive UI**: Clean, modern interface with easy-to-use controls
- **Real-time Recording**: Start, pause/resume, and stop recording with a single click
- **High-Quality Audio**: Records audio at 16kHz sample rate with 16-bit depth
- **File Management**: Automatically saves recordings to a dedicated storage folder
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Lightweight**: Minimal resource usage with no heavy dependencies

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Microphone or audio input device
- Windows, macOS, or Linux

### Installation

1. **Clone or download** this repository to your local machine
2. **Navigate** to the project directory:
   ```bash
   cd quraan
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt file, install the dependencies manually:
   ```bash
   pip install sounddevice numpy
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## How to Use

### Recording Controls

- **Start Button** : Begin recording audio
- **Pause/Resume Button** : Pause or resume the current recording
- **Stop Button** : Stop recording and save the audio file

### Status Indicators

- **⏹️ Idle**: Application is ready but not recording
- **🎙️ Recording...**: Currently recording audio
- **⏸️ Paused**: Recording is paused (can be resumed)
- **✅ Stopped (saved)**: Recording completed and saved

### Audio Output

- **Format**: WAV files (.wav)
- **Sample Rate**: 16,000 Hz (16 kHz)
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit
- **Storage Location**: `storage/` folder in the project directory

## Project Structure

```
quraan/
├── main.py                 # Main application entry point
├── config.py              # Configuration settings
├── recorder/
│   ├── recorder_UI.py     # Tkinter user interface
│   └── recorder_handler.py # Audio recording logic
├── storage/               # Directory for saved audio files
└── README.md             # This file
```

## onfiguration

You can modify the recording settings in `recorder/recorder_handler.py`:

- **Sample Rate**: Change `samplerate` parameter (default: 16000)
- **Channels**: Modify `channels` parameter (default: 1 for mono)
- **Chunk Size**: Adjust `chunk_size` parameter (default: 1024)
- **File Extension**: Change `extension` parameter (default: ".wav")

## Storage

All recorded audio files are automatically saved to the `storage/` directory. The application creates this directory if it doesn't exist. Files are named with the pattern `record.wav` by default.

## Technical Details

- **Audio Library**: Uses `sounddevice` for cross-platform audio input
- **UI Framework**: Built with Tkinter for native look and feel
- **Threading**: Non-blocking audio recording with separate thread
- **File Format**: WAV format for maximum compatibility
- **Error Handling**: Graceful handling of audio device issuesannotated-types==0.7.0