import streamlit as st
import whisper
from moviepy import VideoFileClip  # MoviePy 2.x
import tempfile
import os
import subprocess
import imageio_ffmpeg as ffmpeg

# Load Whisper model (small is faster)
model = whisper.load_model("small")

# Get ffmpeg executable path dynamically
ffmpeg_exe = ffmpeg.get_ffmpeg_exe()

# Streamlit UI
st.title("Video to Text Transcription")
st.write("Upload a video to convert speech to text.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a video file...", type=["mp4", "mov", "avi", "mkv"]
)

def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg and save as WAV."""
    try:
        subprocess.run([
            ffmpeg_exe,
            "-y",  # overwrite automatically
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ], check=True)
    except Exception as e:
        st.error(f"Failed to extract audio: {e}")

if uploaded_file is not None:
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Display video
    st.video(video_path)
    video = VideoFileClip(video_path)

    # Convert to audio WAV file
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    extract_audio_ffmpeg(video_path, audio_path)

    # Transcribe audio
    st.write("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)
    transcription_text = result["text"]

    # Show transcription
    st.subheader("Transcription:")
    st.write(transcription_text)

    # Download button
    st.download_button(
        label="Download Transcription",
        data=transcription_text,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # Clean up
    video.close()
    os.remove(video_path)
    os.remove(audio_path)
