import re
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import subprocess
import json
import os
from faster_whisper import WhisperModel  # Import Faster Whisper

with open('curses.txt', 'r') as curses:
  curses = curses.read()

CURSE_WORDS = curses.split('\n')


def extract_audio(video_path, audio_output):
    """Extract and preprocess audio from video."""
    print("Extracting and preprocessing audio...")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_output)
    clip.close()

    audio = AudioSegment.from_file(audio_output)
    audio = audio.set_channels(1).normalize()
    audio.export(audio_output, format="wav")

def transcribe_audio_with_word_timestamps(audio_path):
    """Transcribe audio and get word-level timestamps using Faster Whisper."""
    print("Transcribing audio with Faster Whisper...")
    # Load Faster Whisper model 
    model = WhisperModel("large-v3", device="auto", compute_type="int8")  # Use "auto" for GPU if available

    # Transcribe with word-level timestamps
    segments, info = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False  # Avoid context carryover
    )

    word_segments = []
    for segment in segments:
        segment_start = segment.start
        segment_end = segment.end
        print(f"Segment: [{segment_start:.2f}s - {segment_end:.2f}s] - {segment.text}")

        for word in segment.words:
            adjusted_start = max(0, word.start - 0.1)  # Shift start earlier by 100ms
            adjusted_end = word.end + 0.1  # Shift end later by 100ms
            word_segments.append({
                'text': word.word,
                'timestamp': (adjusted_start, adjusted_end)
            })
    return word_segments

def blank_audio_word_level(audio_path, words, curse_words):
    """Silence specific words in the audio based on word-level timestamps with a 50ms shift."""
    print("Silencing specified words with a 50ms delay...")
    audio = AudioSegment.from_file(audio_path)

    for word_data in words:
        word = word_data['text'].strip().lower()
        word_cleaned = re.sub(r'[^\w\s]', '', word)  # Remove punctuation

        if any(curse_word in word_cleaned for curse_word in curse_words):
            start_ms = int(word_data['timestamp'][0] * 1000) + 300  # Shift start by 350ms
            end_ms = int(word_data['timestamp'][1] * 1000) + 300    # Shift end by 250ms

            if start_ms > end_ms:
                start_ms = end_ms

            print(f"Silencing '{word_cleaned}' from {start_ms / 1000:.2f}s to {end_ms / 1000:.2f}s")
            audio = audio[:start_ms] + AudioSegment.silent(duration=(end_ms - start_ms)) + audio[end_ms:]
    return audio

def save_word_level_transcript(words, output_path):
    """Save the word-level transcript with timestamps to a file."""
    with open(output_path, "w") as f:
        json.dump(words, f, indent=4)
    print(f"Word-level transcript saved to {output_path}")

def overlay_clean_audio(video_path, clean_audio_path, output_video_path):
    """Overlay cleaned audio back onto the video using FFmpeg."""
    audio_encoders = ["aac", "libmp3lame", "libopus"]
    video_encoders = ["copy", "libx264"]

    for video_codec in video_encoders:
        for audio_codec in audio_encoders:
            temp_output_path = f"{output_video_path.rsplit('.', 1)[0]}_{audio_codec}_{video_codec}.mp4"
            command = [
                "ffmpeg",
                "-i", video_path,
                "-i", clean_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", video_codec,
                "-c:a", audio_codec,
                "-shortest",
                temp_output_path
            ]

            print(f"Trying video codec '{video_codec}' and audio codec '{audio_codec}'...")
            try:
                subprocess.run(command, check=True)
                print(f"Success! Video saved to {temp_output_path}")
                os.rename(temp_output_path, output_video_path)
                return
            except subprocess.CalledProcessError as e:
                print(f"Failed with '{video_codec}' and '{audio_codec}': {e}")
                continue
    raise RuntimeError("Failed to overlay audio with all encoder combinations.")

def cleanup_files(*file_paths):
    """Delete intermediary files."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted intermediary file: {file_path}")

def main(video_path):
    base_name, _ = os.path.splitext(video_path)
    audio_path = f"{base_name}_extracted_audio.wav"
    clean_audio_path = f"{base_name}_clean_audio.wav"
    output_video_path = f"{base_name}_cleaned.mp4"
    transcript_path = f"{base_name}_word_transcript.json"

    try:
        extract_audio(video_path, audio_path)
        word_segments = transcribe_audio_with_word_timestamps(audio_path)
        save_word_level_transcript(word_segments, transcript_path)
        cleaned_audio = blank_audio_word_level(audio_path, word_segments, CURSE_WORDS)
        cleaned_audio.export(clean_audio_path, format="wav")
        overlay_clean_audio(video_path, clean_audio_path, output_video_path)
        print(f"Cleaned video saved to {output_video_path}")
    finally:
        cleanup_files(audio_path, clean_audio_path)

if __name__ == "__main__":
    input_video_path = input("Enter the path to the video file: ")
    main(input_video_path)