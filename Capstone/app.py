import streamlit as st
import whisper
from moviepy import VideoFileClip
import tempfile
import os
import subprocess
import imageio_ffmpeg as ffmpeg

# Ensure Whisper finds ffmpeg
os.environ["FFMPEG_BINARY"] = ffmpeg.get_ffmpeg_exe()

# Load Whisper model
model = whisper.load_model("small")

# Streamlit UI
st.title("Video to Text Transcription")
st.write("Upload a video to convert speech to text.")

# File uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

# Function to extract audio using ffmpeg (Linux-compatible)
def extract_audio(video_path, audio_path):
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()  # Get ffmpeg binary path
    try:
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
    except Exception as e:
        st.error(f"Failed to extract audio: {e}")
        raise

if uploaded_file is not None:
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Display video in Streamlit
    st.video(video_path)

    # Create temporary audio file path
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Extract audio
    st.write("Extracting audio...")
    extract_audio(video_path, audio_path)

    # Transcribe audio
    st.write("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU environments
    transcription = result["text"]

    # Show transcription
    st.subheader("Transcription:")
    st.write(transcription)

    # Download button
    st.download_button(
        label="Download Transcription",
        data=transcription,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # Cleanup temp files
    os.remove(video_path)
    os.remove(audio_path)
