import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import spacy
from collections import Counter
import io
import numpy as np

# 1. KONFIGURATION & NLP
st.set_page_config(page_title="Lore Master Lab", layout="wide")

@st.cache_resource
def load_nlp():
    try:
        return spacy.load("de_core_news_sm")
    except:
        return None

nlp = load_nlp()

# Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { 
        background-color: #1c2128; border: 1px solid #30363d;
        border-radius: 10px; padding: 15px; 
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Literary Intelligence System")
st.subheader("Data Science & Philologische Architektur")

# 2. LORE-PROCESSOR CLASS
class LoreProcessor:
    @staticmethod
    def split_into_chapters(text):
        pattern = r'\n\s*(?:Kapitel|Chapter|Teil|Buch|SECTION|ACT)\s+[0-9IVXLCDM]+|(?:\n\s*[IVXLCDM]+\.\s+)'
        chapters = re.split(pattern, text, flags=re.IGNORECASE)
        return [c.strip() for c in chapters if len(c.strip()) > 200]

    @staticmethod
    def get_metrics(text, selected_names):
        words_raw = re.findall(r'\w+', text.lower())
        tokens_count = len(words_raw)
        unique_words = set(words_raw)
        types_count = len(unique_words)
        
        # Sentiment
        pos_words = ["liebe", "herz", "freude", "licht", "glück", "lachen", "triumph", "strahlend", "hoffnung", "mut"]
        neg_words = ["tod", "schmerz", "angst", "nacht", "weinen", "elend", "verlust", "grab", "einsam", "zorn"]
        sent_score = sum(1 for w in words_raw if w in pos_words) - sum(1 for w in words_raw if w in neg_words)
        normalized_sentiment = (sent_score / tokens_count * 1000) if tokens_count > 0 else 0

        # Herdan's C
        herdan_c = np.log(types_count) / np.log(tokens_count) if tokens_count > 1 else 0

        # Satz-Analyse
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_volatility = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Hapax Legomena & Sternen-Daten
        word_counts = Counter(words_raw)
        hapax_count = sum(1 for word in word_counts if word_counts[word] == 1)
        first_sentence = sentences[0] if sentences else ""
        star_data = [len(w) for w in first_sentence.split()]

        # Charaktere
        char_counts = {name: len(re.findall(rf'\b{name}\b', text, re.IGNORECASE)) for name in selected_names}
        
        return {
            "Sentiment": normalized_sentiment,
            "Lexical_Density_C": herdan_c,
            "Avg_Sentence_Length": avg_sentence_len,
            "Sentence_Volatility": sentence_volatility,
            "Hapax": hapax_count,
            "First_Sentence": first_sentence,
            "Star_Data": star_data,
            "Character_Counts": char_counts
        }

# 3. SIDEBAR
with st.sidebar:
    st.header("📂 Daten-Upload")
    uploaded_file = st.file_uploader("Lade einen Klassiker (.txt)", type="txt")
    st.divider()
    
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        st.header("⚙️ Parameter")
        mode = st.radio("Analyse-Basis:", ["Wort-Blöcke", "Kapitel-Erkennung"])
        if mode == "Wort-Blöcke":
            block_size = st.slider("Wörter pro Block:", 500, 10000, 2000, step=500)
        
        if nlp:
            doc_sample = nlp(raw_text[:30000])
            detected = [ent.text.strip() for ent in doc_sample.ents if ent.label_ == "PER" and len(ent.text) > 2]
            common_names = [name for name, count in Counter(detected).most_common(10)]
            selected_chars = st.multiselect("Figuren:", common_names, default=common_names[:3])
        else:
            selected_chars = []

        theme_color = st.color_picker("Akzentfarbe (Poster)", "#D4AF37")

# 4. HAUPTBEREICH
if uploaded_file:
    # Chunking
    if mode == "Kapitel-Erkennung":
        chunks = LoreProcessor.split_into_chapters(raw_text)
        labels = [f"Kap. {i+1}" for i in range(len(chunks))]
    else:
        all_words = raw_text.split()
        chunks = [" ".join(all_words[i : i + block_size]) for i in range(0, len(all_words), block_size)]
        labels = [f"Block {i+1}" for i in range(len(chunks))]

    # Metriken sammeln
    results = []
    star_maps = []
    for label, chunk in zip(labels, chunks):
        m = LoreProcessor.get_metrics(chunk, selected_chars)
        row = {
            "Abschnitt": label, "Sentiment": m["Sentiment"], 
            "Dichte_C": m["Lexical_Density_C"], "Satzlänge": m["Avg_Sentence_Length"],
            "Volatilität": m["Sentence_Volatility"], "Hapax": m["Hapax"],
            "Erster_Satz": m["First_Sentence"]
        }
        row.update(m["Character_Counts"])
        results.append(row)
        star_maps.append(m["Star_Data"])
    
    df = pd.DataFrame(results)

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Wissenschaftliche Analyse", "👥 Charakter-Netz", "✨ Sternenbild-Poster"])

    with tab1:
        fig_main = make_subplots(rows=2, cols=2, subplot_titles=("Sentiment (normiert)", "Lexikalische Dichte (Herdan C)", "Satzlängen-Rhythmus", "Vokabular-Reichtum (Hapax)"))
        fig_main.add_trace(go.Scatter(x=df['Abschnitt'], y=df['Sentiment'], fill='tozeroy', name="Sentiment", line_color=theme_color), row=1, col=1)
        fig_main.add_trace(go.Scatter(x=df['Abschnitt'], y=df['Dichte_C'], name="Dichte", line_color="#C0C0C0"), row=1, col=2)
        fig_main.add_trace(go.Bar(x=df['Abschnitt'], y=df['Satzlänge'], name="Satzlänge", marker_color="#444444"), row=2, col=1)
        fig_main.add_trace(go.Scatter(x=df['Abschnitt'], y=df['Hapax'], mode='lines+markers', name="Hapax", line_color="#888888"), row=2, col=2)
        fig_main.update_layout(height=700, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_main, use_container_width=True)

    with tab2:
        if selected_chars:
            fig_chars = px.line(df, x="Abschnitt", y=selected_chars, title="Präsenz der Figuren", template="plotly_dark")
            st.plotly_chart(fig_chars, use_container_width=True)
            st.write("#### 🤝 Interaktions-Matrix")
            st.plotly_chart(px.imshow(df[selected_chars].corr(), text_auto=True, color_continuous_scale="Viridis"), use_container_width=True)

    with tab3:
        st.write("### 🌌 Nicolas Rougeaux Edition: Erster-Satz-Konstellationen")
        # Sternenbild-Logik (Polares System)
        fig_stars = go.Figure()
        for i, star_data in enumerate(star_maps):
            if star_data:
                # Winkel berechnen pro Wort
                theta = np.linspace(0, 360, len(star_data), endpoint=False)
                fig_stars.add_trace(go.Scatterpolar(
                    r=star_data, theta=theta, mode='markers+lines',
                    name=df['Abschnitt'].iloc[i],
                    marker=dict(size=star_data, color=theme_color, opacity=0.6)
                ))
        fig_stars.update_layout(polar=dict(bgcolor="#0e1117", radialaxis=dict(visible=False)), showlegend=True, template="plotly_dark", height=800)
        st.plotly_chart(fig_stars, use_container_width=True)

        # Poster Export
        if st.button("🖼️ Poster-Export vorbereiten"):
            buf = io.BytesIO()
            fig_stars.write_image(buf, format="png", scale=2, width=1000, height=1000, engine="kaleido")
            st.session_state.poster = buf.getvalue()
        
        if "poster" in st.session_state:
            st.download_button("📥 Sternenbild herunterladen", data=st.session_state.poster, file_name="sternenbild.png")

else:
    st.info("Bitte lade eine .txt Datei hoch.")
