# AI Text Summariser Dashboard

This project is an AI-powered text summarisation tool built using Streamlit and the FLAN-T5 model. It allows users to summarise long text, website articles, and PDF documents. The application provides multiple summary styles, interactive analytics, and a clean, user-friendly interface designed for students, researchers, and professionals who want quick and meaningful summaries.

---

## Features
- **Multiple Input Sources**
  - Direct text input  
  - Website URLs (auto-scraping of article content)  
  - PDF upload with automatic text extraction  

- **Summary Styles**
  - **Concise** – short and focused  
  - **Detailed** – longer, contextual summary  
  - **Bullet Points** – structured highlights  

- **Smart Text Handling**
  - Automatic text chunking for long documents  
  - Token-aware splitting for FLAN-T5  
  - Hallucination filtering for safe and accurate summaries  

- **Analytics & Visualisation**
  - Word reduction and compression ratio  
  - Keyword extraction  
  - Word cloud generation  
  - Sentence count & character count  
  - Typing animation effect for summary display  

- **UI Enhancements**
  - Modern Streamlit layout  
  - Light/Dark mode toggle  
  - Download summary as TXT file  

---

## Technologies Used
- **Streamlit** – App framework  
- **FLAN-T5 (HuggingFace Transformers)** – Summarisation model  
- **pdfplumber** – Extract text from PDFs  
- **BeautifulSoup4** – Extract text from URLs  
- **WordCloud** – Generate word cloud  
- **Python** – Core logic  

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Aryangandrath/AI-TEXT-SUMMARISER-DASHBOARD.git
cd AI-TEXT-SUMMARISER-DASHBOARD
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run ai.py
```

Then open the local URL displayed in the terminal:
```
http://localhost:8501
```

---

## Live App Link
🔗 **https://ai-text-summariser-dashboard.streamlit.app/**

---

## Use Cases
- Summarising research papers  
- Extracting key points from news articles  
- Summarising PDF documents for study  
- Reducing long text for reports or assignments  
- Creating structured bullet summaries for presentation  

---

## Project Goals
- Build a real, working AI application using FLAN-T5  
- Provide a simple dashboard for everyday summarisation tasks  
- Show how transformers can be used in real UI-based tools  
- Gain hands-on experience with Streamlit deployment  

---

## Future Improvements
- Add OCR support for scanned PDFs  
- Add more summary models (Pegasus, T5-large, BART)  
- Add translation + summarisation combo mode  
- Add summary length slider  
- Optional audio summary output  


---

