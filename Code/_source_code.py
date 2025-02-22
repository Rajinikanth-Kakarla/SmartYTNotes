import json
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
import re
import openai
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

openai.api_key = "YOUR_OPENAI_API_KEY"

languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Telugu': 'te',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Urdu': 'ur',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Odia': 'or',
    'Malayalam': 'ml',
    'Punjabi': 'pa',
}

def load_lottiefile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

summary = load_lottiefile('Lottie/summary.json')
trans = load_lottiefile('Lottie/translate.json')
note = load_lottiefile('Lottie/note.json')

@st.cache_data
def get_video_id(url):
    try:
        video_id = url.split("v=")[1]
        ampersand_position = video_id.find("&")
        if ampersand_position != -1:
            video_id = video_id[:ampersand_position]
        return video_id
    except:
        return None

@st.cache_data
def summarize_text(text):
    if not text:
        return "No text available for summarization."
    summarizer = pipeline('summarization', model="facebook/bart-large-cnn")
    max_allowed_length = min(len(text), 500)
    min_allowed_length = min(30, max_allowed_length - 1)
    num_iters = int(len(text) / 1000)
    sum_text = []
    for i in range(0, num_iters + 1):
        start, end = i * 1000, (i + 1) * 1000
        out = summarizer(text[start:end], max_length=max_allowed_length, min_length=min_allowed_length, do_sample=False)
        sum_text.append(out[0]['summary_text'])
    return " ".join(sum_text)

@st.cache_data
def translate_text(text, target_language):
    if not text:
        return "No text available for translation."
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(text)

def generate_note_making(summary_text):
    if not summary_text:
        return "No text available for note-making."

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a note-making assistant."},
            {"role": "user", "content": summary_text}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

st.title("YouTube Transcript Summarizer 🎥✍️")

yt_video = st.text_input("Enter YouTube Video URL: 🌐")

with st.container():
    col01, col02 = st.columns([2, 2])

    with col01:
        st.subheader("Video 🎥")
        if yt_video:
            st.video(yt_video)

    with col02:
        st.subheader("Transcript from Video 📝")
        video_id = get_video_id(yt_video)
        if video_id:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                result = " ".join([i['text'] for i in transcript])
                cleaned_text = re.sub(r'[^A-Za-z\s]+', '', result)
                st.text_area("Closed Captions", cleaned_text, height=380)
            except Exception as e:
                st.error(f"Error retrieving transcript: {str(e)}")
                cleaned_text = None
        else:
            st.error("Invalid YouTube URL.")
            cleaned_text = None

st.write("___________________________________________________________________________________________")

with st.container():
    col11, col12 = st.columns(2)
    with col11:
        if st.button("Summarize 🤏", key="summarize_button"):
            if cleaned_text:
                st.write("Summarizing...")
                cleaned_summary = summarize_text(cleaned_text)
                st.subheader("Summarized Text:")
                st.text_area('summary', cleaned_summary, height=321)
            else:
                st.error("No text available for summarization.")
    with col12:
        st_lottie(summary, speed=1, key=None)

st.write("___________________________________________________________________________________________")

with st.container():
    col21, col22 = st.columns(2)
    with col21:
        st_lottie(trans, speed=1, key=None)
    with col22:
        selected_language = st.selectbox("Select Language for Translation", list(languages.keys()))
        if st.button("Translate 🗣️", key="translate_button"):
            if cleaned_text:
                st.write(f"Translating to {selected_language}...")
                cleaned_summary = summarize_text(cleaned_text)
                translated_summary = translate_text(cleaned_summary, languages[selected_language])
                st.text_area('translated', translated_summary, height=280)
            else:
                st.error("No text available for translation.")

st.write("___________________________________________________________________________________________")

with st.container():
    col31, col32 = st.columns(2)
    with col31:
        if st.button("Transform 📝", key="transform_button"):
            if cleaned_text:
                st.write("Generating Notes...")
                cleaned_summary = summarize_text(cleaned_text)
                note_making = generate_note_making(cleaned_summary)
                st.subheader("Note-Making:")
                st.text_area('Note-Making', note_making, height=321)
            else:
                st.error("No text available for note-making.")
    with col32:
        st_lottie(note, speed=1, key=None)
