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
    try:
        return spacy.load("de_core_news_sm")
    except:
        return None

nlp = load_nlp()

# Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .stPopover { display: inline-block; vertical-align: middle; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Literary Intelligence System")
st.subheader("Wissenschaftliche Text-Dekomposition")

# 2. LORE-PROCESSOR
class LoreProcessor:
    @staticmethod
    def split_into_chapters(text):
        """Universal-Scanner für Dramen und Prosa (optimiert für Schiller/Gutenberg)."""
        # Erkennt Akte, Szenen, Kapitel und römische Ziffern am Zeilenanfang
        pattern = r'\n\s*(?:(?:Erster|Zweiter|Dritter|Vierter|Fünfter)\s+Akt|(?:Erste|Zweite|Dritte|Vierte|Fünfte)\s+Scene|Kapitel|Chapter|Szene|Akt|Teil)\b|(?:\n\s*[IVXLCDM]+\.\s+\b)'
        chapters = re.split(pattern, text, flags=re.IGNORECASE)
        return [c.strip() for c in chapters if len(c.strip()) > 100]

    @staticmethod
    def get_pos_angle(token):
        """Wortart-Kompass Logik."""
        mapping = {"NOUN": 0, "VERB": 90, "ADJ": 180, "ADV": 270, "PRON": 45, "PROPN": 45, "AUX": 135, "ADP": 225, "DET": 315}
        return mapping.get(token.pos_, 0)

    @staticmethod
    def get_metrics(text, final_characters, top_names_detected):
        if not nlp: return None
        doc = nlp(text[:50000]) # Performance-Cap
        words_raw = [t.text.lower() for t in doc if not t.is_punct]
        tokens_count = len(words_raw)
        if tokens_count < 5: return None
        
        # Sentiment & Dichte
        pos_words = ["liebe", "herz", "freude", "licht", "glück", "lachen", "triumph", "hoffnung", "mut"]
        neg_words = ["tod", "schmerz", "angst", "nacht", "weinen", "elend", "verlust", "grab", "einsam", "zorn"]
        sent_score = sum(1 for w in words_raw if w in pos_words) - sum(1 for w in words_raw if w in neg_words)
        
        # Metriken
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]
        avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Sternen-Daten (Erster Satz)
        first_sent_doc = nlp(sentences[0]) if sentences else []
        star_data = [{"len": len(t.text), "angle": LoreProcessor.get_pos_angle(t), "word": t.text} for t in first_sent_doc if not t.is_punct]

        # Charakter-Präsenz (Bündelung oder alle erkannten)
        char_presence = {}
        target_list = final_characters if final_characters else {n: [n] for n in top_names_detected}
        for final_name, alias_list in target_list.items():
            count = 0
            for alias in alias_list:
                count += len(re.findall(rf'\b{alias}\b', text, re.IGNORECASE))
            char_presence[final_name] = count
        
        return {
            "Sentiment": (sent_score / tokens_count * 1000),
            "Dichte": np.log(len(set(words_raw))) / np.log(tokens_count) if tokens_count > 1 else 0,
            "Satzlänge": avg_sentence_len,
            "Hapax": sum(1 for w, c in Counter(words_raw).items() if c == 1),
            "Star_Data": star_data,
            "Character_Presence": char_presence
        }

# 3. SIDEBAR
with st.sidebar:
    st.header("📂 Daten-Upload")
    uploaded_file = st.file_uploader("Lade .txt hoch", type="txt")
    
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        mode = st.radio("Basis:", ["Kapitel/Szenen", "Wort-Blöcke"])
        
        if mode == "Kapitel/Szenen":
            all_raw_chapters = LoreProcessor.split_into_chapters(raw_text)
            selected_indices = []
            with st.expander("Filter", expanded=True):
                for i, c in enumerate(all_raw_chapters):
                    snippet = c[:30].replace('\n', ' ')
                    if st.checkbox(f"[{i+1}] {snippet}...", value=not any(x in c.lower()[:200] for x in ["copyright", "license"]), key=f"ch_{i}"):
                        selected_indices.append(i)
            chunks = [all_raw_chapters[i] for i in selected_indices]
            labels = [f"Abschnitt {i+1}" for i in selected_indices]
        else:
            size = st.slider("Blockgröße", 500, 5000, 1000)
            words = raw_text.split()
            chunks = [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
            labels = [f"Block {i+1}" for i in range(len(chunks))]

        # CHARAKTER-KOPPLUNG
        st.header("👥 Charaktere")
        doc_sample = nlp(raw_text[:30000]) if nlp else None
        detected = [ent.text.strip() for ent in doc_sample.ents if ent.label_ == "PER" and len(ent.text) > 2] if nlp else []
        top_names = [n for n, c in Counter(detected).most_common(15)]
        
        if 'char_map' not in st.session_state: st.session_state.char_map = {}
        
        with st.expander("Bündelung erstellen"):
            main_name = st.text_input("Hauptname:")
            aliases = st.multiselect("Varianten:", top_names)
            if st.button("Koppeln"):
                st.session_state.char_map[main_name] = aliases
                st.rerun()
        
        if st.session_state.char_map and st.button("Reset Bündelung"):
            st.session_state.char_map = {}; st.rerun()

        theme_color = st.color_picker("Farbe", "#D4AF37")

# 4. HAUPTBEREICH
if uploaded_file and chunks:
    results = []; star_maps_data = []
    active_map = st.session_state.char_map
    
    for label, chunk in zip(labels, chunks):
        m = LoreProcessor.get_metrics(chunk, active_map, top_names)
        if m:
            row = {"Abschnitt": label, **m}
            row.update(m["Character_Presence"])
            results.append(row); star_maps_data.append(m["Star_Data"])
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        t1, t2, t3 = st.tabs(["📊 Dynamik", "👥 Beziehungs-Netz", "✨ Sternenbild"])
        
        with t1:
            st.markdown("#### 📊 Emotionaler Bogen")
            with st.popover("ⓘ Info"):
                st.write("**Sentiment-Analyse:** Misst die emotionale Ladung. Werte > 0 sind positiv, < 0 negativ.")
            st.plotly_chart(px.line(df, x="Abschnitt", y="Sentiment", template="plotly_dark", color_discrete_sequence=[theme_color]), use_container_width=True)
            
            st.divider()
            st.markdown("#### 🧠 Lexikalische Dichte")
            with st.popover("ⓘ Info"):
                st.write("**Herdan's C:** Misst den Wortschatz-Reichtum. Ein Wert von 1.0 wäre maximale Vielfalt.")
            st.plotly_chart(px.line(df, x="Abschnitt", y="Dichte", template="plotly_dark", color_discrete_sequence=["#C0C0C0"]), use_container_width=True)

        with t2:
            display_chars = list(active_map.keys()) if active_map else top_names[:5]
            st.plotly_chart(px.line(df, x="Abschnitt", y=display_chars, title="Präsenz", template="plotly_dark"), use_container_width=True)
            st.plotly_chart(px.imshow(df[display_chars].corr(), text_auto=True, color_continuous_scale="Viridis", title="Interaktion"), use_container_width=True)

        with t3:
            with st.popover("ⓘ Kompass & Anleitung"):
                st.write("**Wortart-Kompass:**")
                st.write("⬆️ 0°: Nomen | ➡️ 90°: Verben | ⬇️ 180°: Adjektive | ⬅️ 270°: Adverbien")
                st.write("**Radius:** Wortlänge.")
            
            variant = st.radio("Typ:", ["Urknall", "Sonnensystem"])
            fig = go.Figure()
            
            if variant == "Urknall":
                colors = px.colors.sample_colorscale("Viridis", len(star_maps_data))
                for i, data in enumerate(star_maps_data):
                    if data:
                        fig.add_trace(go.Scatterpolar(r=[d['len'] for d in data], theta=[d['angle'] for d in data], mode='markers+lines', name=labels[i], line_color=colors[i]))
            else:
                orbit_shift = np.linspace(0, 360, len(star_maps_data), endpoint=False)
                colors = px.colors.sample_colorscale("Plasma", len(star_maps_data))
                for i, data in enumerate(star_maps_data):
                    if data:
                        theta = np.linspace(0, 360, len(data), endpoint=False) + orbit_shift[i]
                        fig.add_trace(go.Scatterpolar(r=np.add([d['len'] for d in data], 20), theta=theta, mode='markers+lines', name=labels[i], line_color=colors[i]))
            
            fig.update_layout(height=800, template="plotly_dark", polar=dict(radialaxis=dict(visible=False), angularaxis=dict(tickvals=[0, 90, 180, 270], ticktext=["Nomen", "Verb", "Adj", "Adv"])))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Keine Daten zur Anzeige. Bitte Kapitel-Filter prüfen.")
