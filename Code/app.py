# Imports
import json
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re
import openai
from streamlit_lottie import st_lottie
from transformers import MarianMTModel, MarianTokenizer

st.set_page_config(layout="wide")

openai.api_key = "YOUR_OPENAI_API_KEY"

class LottieLoader:
    @staticmethod
    def load_lottiefile(filepath: str):
        with open(filepath, 'r') as f:
            return json.load(f)

summary = LottieLoader.load_lottiefile('Lottie/summary.json')
trans = LottieLoader.load_lottiefile('Lottie/translate.json')
note = LottieLoader.load_lottiefile('Lottie/note.json')

class VideoIDExtractor:
    @staticmethod
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

class TextSummarizer:
    @staticmethod
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

class TextTranslator:
    def __init__(self, target_language):
        self.target_language = target_language
        self.model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate_text(self, text):
        if not text:
            return "No text available for translation."
        inputs = self.tokenizer.encode(text, return_tensors="pt")
        translated = self.model.generate(inputs, max_length=1000)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

def generate_note_making(summary_text):
    if not summary_text:
        return "No text available for note-making."

    client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

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

class YouTubeTranscriptSummarizerApp:
    def __init__(self):
        self.yt_video = st.text_input("Enter YouTube Video URL: 🌐")
        self.cleaned_text = None

    def run(self):
        st.title("YouTube Transcript Summarizer 🎥✍️")

        with st.container():
            col01, col02 = st.columns([2, 2])

            with col01:
                st.subheader("Video 🎥")
                if self.yt_video:
                    st.video(self.yt_video)

            with col02:
                st.subheader("Transcript from Video 📝")
                video_id = VideoIDExtractor.get_video_id(self.yt_video)
                if video_id:
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        result = " ".join([i['text'] for i in transcript])
                        self.cleaned_text = re.sub(r'[^A-Za-z\s]+', '', result)
                        st.text_area("Closed Captions", self.cleaned_text, height=380)
                    except Exception as e:
                        st.error(f"Error retrieving transcript: {str(e)}")
                        self.cleaned_text = None
                else:
                    st.error("Invalid YouTube URL.")
                    self.cleaned_text = None

        st.write("___________________________________________________________________________________________")

        with st.container():
            col11, col12 = st.columns(2)
            with col11:
                if st.button("Summarize 🤏", key="summarize_button"):
                    if self.cleaned_text:
                        st.write("Summarizing...")
                        cleaned_summary = TextSummarizer.summarize_text(self.cleaned_text)
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
                selected_language = st.selectbox("Select Language for Translation", ['hi', 'de', 'es', 'it', 'ru', 'nl', 'zh', 'ja', 'ar', 'fr'])
                if st.button("Translate 🗣️", key="translate_button"):
                    if self.cleaned_text:
                        st.write(f"Translating to {selected_language.upper()}...")
                        cleaned_summary = TextSummarizer.summarize_text(self.cleaned_text)
                        translator = TextTranslator(selected_language)
                        translated_summary = translator.translate_text(cleaned_summary)
                        st.text_area('translated', translated_summary, height=280)
                    else:
                        st.error("No text available for translation.")

        st.write("___________________________________________________________________________________________")

        with st.container():
            col31, col32 = st.columns(2)
            with col31:
                if st.button("Transform 📝", key="transform_button"):
                    if self.cleaned_text:
                        st.write("Generating Notes...")
                        cleaned_summary = TextSummarizer.summarize_text(self.cleaned_text)
                        note_making = generate_note_making(cleaned_summary)
                        st.subheader("Note-Making:")
                        st.text_area('Note-Making', note_making, height=321)
                    else:
                        st.error("No text available for note-making.")
            with col32:
                st_lottie(note, speed=1, key=None)

if __name__ == "__main__":
    app = YouTubeTranscriptSummarizerApp()
    app.run()
