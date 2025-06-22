import os
import tempfile
import re
from collections import Counter
import moviepy.editor as mp
import speech_recognition as sr
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from tkinter import Tk, filedialog

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Summarizer model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extract_keywords(text, top_n=5):
    words = re.findall(r'\w+', text.lower())
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    return [word for word, _ in Counter(filtered).most_common(top_n)]


def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    keywords = extract_keywords(text)
    return summary, keywords


def pick_file(filetypes):
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    return filepath


def summarize_article():
    print("\n📄 Select a text file...")
    filepath = pick_file([("Text Files", "*.txt")])
    if not filepath:
        print("No file selected.")
        return
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    summary, keywords = summarize_text(content)
    print("\n=== SUMMARY ===")
    print(summary)
    print("\n=== KEYWORDS ===")
    print(", ".join(keywords))


def summarize_video():
    print("\n🎥 Select a video file...")
    filepath = pick_file([("Video Files", "*.mp4 *.mov *.avi *.mkv")])
    if not filepath:
        print("No file selected.")
        return
    print(f"Processing: {filepath}")

    # Extract audio
    video = mp.VideoFileClip(filepath)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video.audio.write_audiofile(temp_audio, verbose=False, logger=None)

    # Transcribe
    r = sr.Recognizer()
    with sr.AudioFile(temp_audio) as source:
        audio = r.record(source)
    os.remove(temp_audio)

    try:
        transcript = r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Could not transcribe audio.")
        return

    summary, keywords = summarize_text(transcript)
    print("\n=== TRANSCRIPT (start) ===")
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))
    print("\n=== SUMMARY ===")
    print(summary)
    print("\n=== KEYWORDS ===")
    print(", ".join(keywords))


def main():
    print("== AI Content Summarizer ==")
    while True:
        print("\n1. Summarize Article (TXT File)")
        print("2. Summarize Video (MP4 etc.)")
        print("0. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            summarize_article()
        elif choice == "2":
            summarize_video()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
