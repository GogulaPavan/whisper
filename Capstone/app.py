import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
import tempfile
import os
import subprocess
import imageio_ffmpeg as ffmpeg

# Load Whisper model
st.title("Video to Text Transcription")
st.write("Upload a video and convert speech to text.")

model = whisper.load_model("small")

# Get ffmpeg executable path
ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg"""
    subprocess.run([
        ffmpeg_exe,
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ], check=True)

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4","mov","avi","mkv"])

if uploaded_file:
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)
    video = VideoFileClip(video_path)

    # Extract audio to temp WAV file
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    extract_audio(video_path, audio_path)

    st.write("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)
    transcription = result["text"]

    st.subheader("Transcription:")
    st.write(transcription)

    st.download_button(
        "Download Transcription",
        transcription,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # Cleanup
    video.close()
    os.remove(video_path)
    os.remove(audio_path)
