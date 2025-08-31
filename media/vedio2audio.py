from moviepy.editor import VideoFileClip

def video_to_audio(video_path, audio_path):
    video = VideoFileClip(video_path)    
    audio = video.audio    
    audio.write_audiofile(audio_path)

video_to_audio("AlAalaq.mp4", "AlAalaq.wav")
