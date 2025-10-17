import streamlit as st
import whisper
from moviepy.editor import VideoFileClip
import tempfile
import os
import subprocess

# Load the Whisper model (small or medium model for faster processing)
model = whisper.load_model("small")

# Set the FFMPEG_PATH environment variable explicitly
os.environ["FFMPEG_PATH"] = r"C:\Users\gogul\Downloads\ffmpeg-2024-11-03-git-df00705e00-full_build\ffmpeg-2024-11-03-git-df00705e00-full_build\bin\ffmpeg.exe"
# Also add the path to the system PATH to ensure subprocess access
os.environ["PATH"] += os.pathsep + r"C:\Users\gogul\Downloads\ffmpeg-2024-11-03-git-df00705e00-full_build\ffmpeg-2024-11-03-git-df00705e00-full_build\bin"

# Streamlit UI setup
st.title("Video to Text Transcription")
st.write("Upload a video to convert speech to text.")
    
# File uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

# Custom function to extract audio using ffmpeg directly
def extract_audio_ffmpeg(video_path, audio_path):
    try:
        # Run ffmpeg command to convert video to audio
        subprocess.run([
            os.environ["FFMPEG_PATH"],
             "-y",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
        ], check=True)
    except Exception as e:
        st.error(f"Failed to extract audio: {e}")

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Load video file using the temporary path
    st.video(video_path)
    video = VideoFileClip(video_path)

    # Convert video to audio using ffmpeg and save as a temporary .wav file
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    extract_audio_ffmpeg(video_path, audio_path)

    # Transcribe audio to text using Whisper (specifying ffmpeg path)
    st.write("Transcribing audio...")
    result = model.transcribe(audio_path, fp16=False)  # Add fp16=False if necessary
    transcription_text = result['text']

    # Display transcription
    st.subheader("Transcription:")
    st.write(transcription_text)

    # Option to download the transcription
    st.download_button(
        label="Download Transcription",
        data=transcription_text,
        file_name="transcription.txt",
        mime="text/plain"
    )

    # Clean up temporary files
    video.close()  # Explicitly close the video file
    os.remove(video_path)  # Delete the video file after closing it
    os.remove(audio_path)  # Delete the audio file