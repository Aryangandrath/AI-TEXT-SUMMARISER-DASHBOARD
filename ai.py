# ai.py  (main Streamlit app)

import time
from collections import Counter

import requests
import streamlit as st
import textstat
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer
from wordcloud import WordCloud
import pdfplumber


# ------------------ LOAD SUMMARIZATION MODEL (FLAN-T5) ------------------ #
@st.cache_resource
def load_model():
    """
    Load the FLAN-T5 model wrapped in a HuggingFace pipeline.
    FLAN-T5 respects length controls better than BART.
    """
    return pipeline("summarization", model="google/flan-t5-large")


summarizer = load_model()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


# ------------------ STREAMLIT CONFIG ------------------ #
st.set_page_config(
    page_title="AI Text Summariser Dashboard",
    layout="wide",
    page_icon="📊"
)

# ------------------ SIDEBAR ------------------ #
st.sidebar.title("⚙️ Options")
input_type = st.sidebar.radio(
    "Choose Input Type:",
    ["✍️ Text", "🌐 URL", "📄 PDF Upload"]
)
style = st.sidebar.selectbox(
    "Summary Style:",
    ["Concise", "Detailed", "Bullet Points"]
)
theme = st.sidebar.checkbox("🌙 Dark Mode")
st.sidebar.markdown("---")
st.sidebar.info("Powered by FLAN-T5 + Streamlit")

# ------------------ DARK MODE CSS ------------------ #
if theme:
    st.markdown(
        """
        <style>
        .reportview-container, .sidebar-content {background-color: #0e1117; color: #fafafa;}
        .stTextArea textarea, .stTextInput input {background-color: #262730; color: #fafafa;}
        .stButton>button {background-color:#1abc9c; color:#000;}
        .stSelectbox, .stRadio {color: #fafafa;}
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------ MAIN TITLE ------------------ #
st.title("📊 AI Text Summariser Dashboard")
st.write("Classic, powerful & elegant way to shorten long texts.")

# ------------------ INPUT SECTION ------------------ #
text = ""

if input_type == "✍️ Text":
    text = st.text_area("Paste your text here:", height=200)

elif input_type == "🌐 URL":
    url = st.text_input("Enter article URL:")
    if url:
        try:
            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")
            # Extract text from all <p> tags
            text = " ".join(
                [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            )
            word_count = len(text.split())
            if word_count > 0:
                st.success(
                    f"✅ Article fetched successfully! Extracted ~{word_count} words."
                )
            else:
                st.warning(
                    "⚠️ Article fetched, but extracted very little text. "
                    "The site may be JS-heavy or protected."
                )
        except Exception as e:
            st.error(f"❌ Failed to fetch article. Error: {e}")

elif input_type == "📄 PDF Upload":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        try:
            text_pages = []
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text() or ""
                    text_pages.append(extracted)
            text = "\n".join(text_pages)
            word_count = len(text.split())
            if word_count > 0:
                st.success(
                    f"✅ PDF text extracted successfully! Extracted ~{word_count} words."
                )
            else:
                st.warning(
                    "⚠️ PDF opened, but extracted very little text. "
                    "It may be scanned images (needs OCR)."
                )
        except Exception as e:
            st.error(f"❌ Failed to read PDF. Error: {e}")

# Optional debug/info
if text:
    st.caption(f"Loaded ~{len(text.split())} words from input.")


# ------------------ CHUNKING FUNCTION (WORD-BASED, TOKEN-AWARE) ------------------ #
def chunk_text(full_text: str, max_tokens: int = 900):
    """
    Split the full text into chunks whose token length
    (using FLAN-T5 tokenizer) stays under `max_tokens`,
    while preserving *all* words.
    """
    words = full_text.split()
    chunks = []
    current_words = []

    for w in words:
        tentative = " ".join(current_words + [w])
        token_count = len(tokenizer.encode(tentative))

        if token_count > max_tokens:
            if current_words:
                chunks.append(" ".join(current_words))
            current_words = [w]
        else:
            current_words.append(w)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ------------------ HALLUCINATION / NOISE CLEANER ------------------ #
def clean_summary(summary_text: str, original_text: str) -> str:
    """
    Remove clearly unrelated / risky sentences (e.g., helplines)
    and sentences with very low lexical overlap with the source.
    """
    bad_phrases = [
        "suicide",
        "samaritan",
        "samaritans",
        "self-harm",
        "crisis hotline",
        "helpline",
    ]

    orig_words = set(original_text.lower().split())
    cleaned_sentences = []

    # Rough sentence split
    raw_sentences = summary_text.replace("\n", " ").split(". ")
    for raw in raw_sentences:
        s = raw.strip()
        if not s:
            continue

        s_lower = s.lower()

        # Filter obviously inappropriate boilerplate
        if any(bp in s_lower for bp in bad_phrases):
            continue

        # Lexical overlap filter
        words = set(s_lower.split())
        overlap = len(words & orig_words) / max(1, len(words))
        if overlap < 0.15:
            # Very little overlap → likely hallucinated / off-topic
            continue

        if not s.endswith("."):
            s += "."
        cleaned_sentences.append(s)

    return " ".join(cleaned_sentences)


# ------------------ SUMMARY GENERATION ------------------ #
if st.button("🚀 Generate Summary"):
    if text.strip() == "":
        st.warning("⚠️ Please provide some input text.")
    else:
        orig_words_count = len(text.split())

        # Progress bar UX
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            progress_bar.progress(i)
            status_text.text(f"Processing... {i}%")
            time.sleep(0.01)
        status_text.text("Generating summary...")
        time.sleep(0.2)

        # Token-aware chunking
        chunks = chunk_text(text)
        st.caption(f"Chunked into {len(chunks)} part(s) for summarization.")

        all_summaries = []

        for idx, chunk in enumerate(chunks, start=1):
            words_in_chunk = len(chunk.split())

            # --- Length control (Concise vs Detailed vs Bullet) ---
            if style == "Concise":
                # ~30–40% of chunk length, within safe bounds
                max_len = min(160, max(60, int(words_in_chunk * 0.4)))
                min_len = min(max_len - 10, max(30, int(words_in_chunk * 0.2)))

            elif style == "Detailed":
                # ~60–80% of chunk length, longer than Concise
                max_len = min(360, max(180, int(words_in_chunk * 0.8)))
                min_len = min(max_len - 20, max(120, int(words_in_chunk * 0.4)))

            else:  # Bullet Points
                # Medium-long summary suitable to split into bullets
                max_len = min(300, max(150, int(words_in_chunk * 0.6)))
                min_len = min(max_len - 20, max(80, int(words_in_chunk * 0.3)))

            # Safety: ensure min_len < max_len
            if min_len >= max_len:
                min_len = max_len // 2

            summary_chunk = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                length_penalty=0.1,      # encourage longer outputs
                no_repeat_ngram_size=3,
                do_sample=False,
            )

            all_summaries.append(summary_chunk[0]["summary_text"])

        # Combine chunk summaries
        summary_text = " ".join(all_summaries)

        # Clean hallucinations / unrelated boilerplate
        summary_text = clean_summary(summary_text, text)

        # Ordered bullet points if selected
        if style == "Bullet Points":
            sentences = summary_text.split(". ")
            bullet_lines = []
            for i, sentence in enumerate(sentences, 1):
                sentence = sentence.strip()
                if sentence:
                    if not sentence.endswith("."):
                        sentence += "."
                    bullet_lines.append(f"{i}. {sentence}")
            summary_text = "\n".join(bullet_lines)

        # Typing animation effect
        display_text = ""
        summary_area = st.empty()
        for word in summary_text.split():
            display_text += word + " "
            summary_area.text(display_text)
            time.sleep(0.01)

        # ------------------ RESULTS ------------------ #
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📜 Original Text")
            st.text_area("Original", text, height=300)

        with col2:
            st.subheader("✨ AI Summary")
            st.success(summary_text)

        # ------------------ ANALYTICS ------------------ #
        st.markdown("---")
        st.subheader("📊 Text Analytics")
        summ_len = len(summary_text.split())
        ratio = round(summ_len / orig_words_count * 100, 2) if orig_words_count > 0 else 0

        met1, met2, met3 = st.columns(3)
        met1.metric("Original Length", f"{orig_words_count} words")
        met2.metric("Summary Length", f"{summ_len} words")
        met3.metric("Compression", f"{ratio}%")

        # ------------------ KEYWORDS + WORDCLOUD ------------------ #
        st.markdown("### 🔑 Top Keywords")
        words_only = [w.lower() for w in summary_text.split() if len(w) > 3]
        common_words = Counter(words_only).most_common(10)
        if common_words:
            st.write(", ".join([w[0] for w in common_words]))
        else:
            st.write("Not enough text to extract keywords.")

        st.markdown("### 🌐 Word Cloud")
        if summary_text.strip():
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(summary_text)
            st.image(wordcloud.to_array())
        else:
            st.write("No summary text available for word cloud.")

        st.markdown("### 📖 Readability & Stats")
        if summary_text.strip():
            try:
                grade_level = textstat.flesch_kincaid_grade(summary_text)
                st.write(f"**Flesch-Kincaid Grade Level:** {grade_level:.2f}")
            except Exception as e:
                st.write(f"Could not compute readability: {e}")
            st.write(f"**Number of Sentences:** {len(summary_text.split('. '))}")
            st.write(f"**Number of Characters:** {len(summary_text)}")
        else:
            st.write("No summary text available for readability analysis.")

        # ------------------ DOWNLOAD ------------------ #
        st.markdown("---")
        st.download_button(
            "⬇️ Download Summary as TXT",
            summary_text,
            file_name="summary.txt"
        )
