import streamlit as st
import pandas as pd
import nltk
import os
from nltk.tokenize import sent_tokenize
from faster_whisper import WhisperModel

# -------------------- NLTK SETUP --------------------
nltk.download("punkt")

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="Audio Interview Q&A Extractor",
    layout="centered"
)

st.title("🎤 Audio Interview Q&A Extractor")
st.write("Upload interview audio → get structured Question & Answer table")

# -------------------- FILE UPLOAD --------------------
audio_file = st.file_uploader(
    "Upload interview audio file",
    type=["mp3", "wav", "m4a"]
)

if audio_file is not None:
    # Save uploaded audio
    os.makedirs("temp_audio", exist_ok=True)
    audio_path = os.path.join("temp_audio", audio_file.name)

    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio(audio_file)

    # -------------------- TRANSCRIPTION --------------------
    st.info("⏳ Transcribing audio… please wait")

    # ✅ FAST & CLOUD-SAFE MODEL
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(audio_path)

    raw_text = ""
    for segment in segments:
        raw_text += segment.text + " "

    st.success("✅ Transcription completed")

    # -------------------- SHOW TRANSCRIPT --------------------
    st.subheader("📝 Raw Transcript")
    st.text_area("Transcript", raw_text, height=250)

    # -------------------- Q&A EXTRACTION --------------------
    st.info("🔍 Extracting Questions & Answers")

    sentences = sent_tokenize(raw_text)

    questions = []
    answers = []
    current_question = ""

    for sent in sentences:
        sent = sent.strip()

        if len(sent) < 3:
            continue

        if (
            sent.lower().startswith(
                ("what", "why", "how", "when", "where", "who", "can", "do", "does")
            )
            or "?" in sent
        ):
            current_question = sent
        else:
            if current_question:
                questions.append(current_question)
                answers.append(sent)
                current_question = ""

    # -------------------- OUTPUT --------------------
    if len(questions) == 0:
        st.warning("⚠️ No clear Q&A detected. Audio may be monologue-based.")
    else:
        df = pd.DataFrame({
            "Question": questions,
            "Answer": answers
        })

        st.subheader("📊 Extracted Q&A")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Q&A as CSV",
            data=csv,
            file_name="interview_qa.csv",
            mime="text/csv"
        )

