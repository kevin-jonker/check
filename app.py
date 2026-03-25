import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import spacy
from collections import Counter
import io
import numpy as np

# 1. KONFIGURATION & NLP
st.set_page_config(page_title="Lore Master Lab", layout="wide")

@st.cache_resource
def load_nlp():
    with st.spinner("Lade Sprachmodell..."):
        try:
            return spacy.load("de_core_news_sm")
        except Exception as e:
            st.error(f"Fehler beim Laden von spaCy: {e}")
            return None

nlp = load_nlp()

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { 
        background-color: #1c2128; border: 1px solid #30363d;
        border-radius: 10px; padding: 15px; 
    }
    .stPopover { display: inline-block; margin-left: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Literary Intelligence System")
st.subheader("Data Science & Narrative DNA")

# 2. LORE-PROCESSOR
class LoreProcessor:
    @staticmethod
    def split_into_chapters(text):
        pattern = r'\n\s*(?:Kapitel|Chapter|Teil|Buch|SECTION|ACT)\s+[0-9IVXLCDM]+\b|(?:\n\s*[IVXLCDM]+\.\s+\b)'
        chapters = re.split(pattern, text, flags=re.IGNORECASE)
        return [c.strip() for c in chapters if len(c.strip()) > 150]

    @staticmethod
    def get_metrics(text, final_characters):
        words_raw = re.findall(r'\w+', text.lower())
        tokens_count = len(words_raw)
        if tokens_count < 10: return None
            
        unique_words = set(words_raw)
        types_count = len(unique_words)
        
        pos_words = ["liebe", "herz", "freude", "licht", "glück", "lachen", "triumph", "strahlend", "hoffnung", "mut"]
        neg_words = ["tod", "schmerz", "angst", "nacht", "weinen", "elend", "verlust", "grab", "einsam", "zorn"]
        sent_score = sum(1 for w in words_raw if w in pos_words) - sum(1 for w in words_raw if w in neg_words)
        normalized_sentiment = (sent_score / tokens_count * 1000)

        herdan_c = np.log(types_count) / np.log(tokens_count) if tokens_count > 1 else 0

        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_volatility = np.std(sentence_lengths) if sentence_lengths else 0
        
        word_counts = Counter(words_raw)
        hapax_count = sum(1 for word in word_counts if word_counts[word] == 1)
        first_sentence = sentences[0] if sentences else ""
        star_data = [len(w) for w in first_sentence.split()]

        char_presence = {}
        for final_name, alias_list in final_characters.items():
            count = 0
            for alias in alias_list:
                count += len(re.findall(rf'\b{alias}\b', text, re.IGNORECASE))
            char_presence[final_name] = count
        
        return {
            "Sentiment": normalized_sentiment, "Lexical_Density_C": herdan_c,
            "Avg_Sentence_Length": avg_sentence_len, "Sentence_Volatility": sentence_volatility,
            "Hapax": hapax_count, "First_Sentence": first_sentence,
            "Star_Data": star_data, "Character_Presence": char_presence
        }

# 3. SIDEBAR
with st.sidebar:
    st.header("📂 Daten-Upload")
    uploaded_file = st.file_uploader("Lade einen Klassiker (.txt)", type="txt")
    st.divider()
    
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        st.header("⚙️ Analyse-Settings")
        mode = st.radio("Analyse-Basis:", ["Wort-Blöcke", "Kapitel-Erkennung"])
        
        if mode == "Kapitel-Erkennung":
            all_raw_chapters = LoreProcessor.split_into_chapters(raw_text)
            st.write("### ✂️ Kapitel-Filter")
            selected_indices = []
            for i, c in enumerate(all_raw_chapters):
                snippet = c[:40].replace('\n', ' ')
                is_junk = any(x in c.lower()[:500] for x in ["copyright", "gutenberg", "license", "trademarks"])
                if st.checkbox(f"[{i+1}] {snippet}...", value=not is_junk, key=f"ch_{i}"):
                    selected_indices.append(i)
            chunks = [all_raw_chapters[i] for i in selected_indices]
            labels = [f"Kap. {i+1}" for i in selected_indices]
        else:
            block_size = st.slider("Wörter pro Block:", 500, 10000, 2000, step=500)
            all_words = raw_text.split()
            chunks = [" ".join(all_words[i : i + block_size]) for i in range(0, len(all_words), block_size)]
            labels = [f"Block {i+1}" for i in range(len(chunks))]

        st.header("👥 Charakter-Management")
        if nlp:
            with st.spinner("Extrahiere Figuren..."):
                doc_sample = nlp(raw_text[:40000])
                detected = [ent.text.strip() for ent in doc_sample.ents if ent.label_ == "PER" and len(ent.text) > 2]
                top_names = [name for name, count in Counter(detected).most_common(12)]
        else: top_names = []
            
        bundle_input = st.text_area("Bündelungs-Regeln:", height=100, value="Gregor, Gregors, Samsa -> Gregor")
        final_characters = {}
        mapped_aliases = set()
        for rule in bundle_input.split('\n'):
            if "->" in rule:
                aliases_part, final_name = rule.split('->')
                final_name = final_name.strip()
                alias_list = [a.strip() for a in aliases_part.split(',') if a.strip()]
                final_characters[final_name] = alias_list
                for a in alias_list: mapped_aliases.add(a)
        
        for name in top_names:
            if name not in mapped_aliases and name not in final_characters:
                final_characters[name] = [name]
        
        selected_final_chars = st.multiselect("Figuren wählen:", list(final_characters.keys()), default=list(final_characters.keys())[:3])
        theme_color = st.color_picker("Akzentfarbe", "#D4AF37")

# 4. HAUPTBEREICH
if uploaded_file and 'chunks' in locals():
    results = []
    star_maps = []
    active_chars = {k: v for k, v in final_characters.items() if k in selected_final_chars}
    
    progress_bar = st.progress(0, "Analysiere Text...")
    for i, (label, chunk) in enumerate(zip(labels, chunks)):
        m = LoreProcessor.get_metrics(chunk, active_chars)
        if m:
            row = {
                "Abschnitt": label, "Sentiment": m["Sentiment"], 
                "Dichte_C": m["Lexical_Density_C"], "Satzlänge": m["Avg_Sentence_Length"],
                "Volatilität": m["Sentence_Volatility"], "Hapax": m["Hapax"],
                "Erster_Satz": m["First_Sentence"]
            }
            for name, count in m["Character_Presence"].items(): row[name] = count
            results.append(row)
            star_maps.append(m["Star_Data"])
        progress_bar.progress((i+1) / len(chunks))
    progress_bar.empty()
    
    df = pd.DataFrame(results)
    tab1, tab2, tab3 = st.tabs(["📊 Narrative Dynamik", "👥 Beziehungs-Netz", "✨ Sternenbild-Poster"])

    with tab1:
        st.write("#### 📊 Emotionaler Bogen")
        fig_sent = px.line(df, x="Abschnitt", y="Sentiment", template="plotly_dark", color_discrete_sequence=[theme_color])
        fig_sent.update_layout(yaxis=dict(title="Score (Normiert)", zeroline=True, zerolinecolor='white'))
        st.plotly_chart(fig_sent, use_container_width=True)
        
        st.divider()
        st.write("#### 🧠 Lexikalische Dichte (Herdan's C)")
        fig_dens = px.line(df, x="Abschnitt", y="Dichte_C", template="plotly_dark", color_discrete_sequence=["#C0C0C0"])
        st.plotly_chart(fig_dens, use_container_width=True)
        with st.popover("ⓘ Info zur Dichte"):
            st.write(r"**Methode:** $C = \frac{\log(Unique)}{\log(Total)}$")

        st.divider()
        st.write("#### ⏱️ Satzlängen-Rhythmus (Puls)")
        fig_len = px.bar(df, x="Abschnitt", y="Satzlänge", template="plotly_dark", color_discrete_sequence=["#444444"])
        st.plotly_chart(fig_len, use_container_width=True)

        st.divider()
        st.write("#### ✨ Vokabular-Reichtum")
        fig_hapax = px.line(df, x="Abschnitt", y="Hapax", template="plotly_dark", color_discrete_sequence=["#888888"])
        st.plotly_chart(fig_hapax, use_container_width=True)

    with tab2:
        if selected_final_chars:
            st.write("#### 👤 Präsenz der Figuren")
            fig_chars = px.line(df, x="Abschnitt", y=selected_final_chars, template="plotly_dark")
            st.plotly_chart(fig_chars, use_container_width=True)
            
            st.divider()
            st.write("#### 🤝 Interaktions-Matrix")
            corr = df[selected_final_chars].corr()
            fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis")
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.write("### 🌌 Nicolas Rougeaux Edition")
        variant = st.radio("Variante:", ["Urknall", "Sonnensystem"])
        fig_stars = go.Figure()
        
        if variant == "Urknall":
            colors = px.colors.sample_colorscale("Viridis", len(star_maps))
            for i, data in enumerate(star_maps):
                if data:
                    theta = np.linspace(0, 360, len(data), endpoint=False)
                    fig_stars.add_trace(go.Scatterpolar(r=data, theta=theta, mode='markers+lines', name=df['Abschnitt'].iloc[i], marker=dict(color=colors[i])))
        else:
            for i, data in enumerate(star_maps):
                if data:
                    fig_stars.add_trace(go.Scatterpolar(r=np.add(data, 10), theta=np.linspace(0, 360, len(data), endpoint=False), mode='markers+lines', marker=dict(color=theme_color)))

        fig_stars.update_layout(height=800, template="plotly_dark")
        st.plotly_chart(fig_stars, use_container_width=True)
else:
    st.info("Bitte lade eine .txt Datei hoch.")
