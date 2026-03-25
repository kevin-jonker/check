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
            st.error(f"Fehler: {e}")
            return None

nlp = load_nlp()

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { background-color: #1c2128; border: 1px solid #30363d; border-radius: 10px; padding: 15px; }
    .stPopover { display: inline-block; margin-left: 5px; vertical-align: middle; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Literary Intelligence System")
st.subheader("Wissenschaftliche Text-Dekomposition & Data Art")

# 2. LORE-PROCESSOR (Logik-Einheit)
class LoreProcessor:
    @staticmethod
    def split_into_chapters(text):
        """Erkennt Kapitel, Akte, Szenen oder Abschnitte (inkl. Dramen-Struktur)."""
        pattern = r'\n\s*(?:Kapitel|Chapter|Teil|Buch|SECTION|ACT|Akt|Scene|Szene)\s+[0-9IVXLCDM]+\b|(?:\n\s*[IVXLCDM]+\.\s+\b)'
        chapters = re.split(pattern, text, flags=re.IGNORECASE)
        # Filter: Muss Inhalt haben
        return [c.strip() for c in chapters if len(c.strip()) > 100]

    @staticmethod
    def get_pos_angle(token):
        """Ordnet Wortarten spezifische Winkel zu für die Urknall-Visualisierung."""
        mapping = {
            "NOUN": 0,      # Norden
            "VERB": 90,     # Osten
            "ADJ": 180,     # Süden
            "ADV": 270,     # Westen
            "PRON": 45, "PROPN": 45,
            "AUX": 135, "ADP": 225, "DET": 315
        }
        return mapping.get(token.pos_, 0)

    @staticmethod
    def get_metrics(text, final_characters):
        if not nlp: return None
        doc = nlp(text)
        words_raw = [t.text.lower() for t in doc if not t.is_punct]
        tokens_count = len(words_raw)
        if tokens_count < 10: return None
            
        unique_words = set(words_raw)
        
        # Sentiment
        pos_words = ["liebe", "herz", "freude", "licht", "glück", "lachen", "triumph", "hoffnung", "mut"]
        neg_words = ["tod", "schmerz", "angst", "nacht", "weinen", "elend", "verlust", "grab", "einsam", "zorn"]
        sent_score = sum(1 for w in words_raw if w in pos_words) - sum(1 for w in words_raw if w in neg_words)
        normalized_sentiment = (sent_score / tokens_count * 1000)

        herdan_c = np.log(len(unique_words)) / np.log(tokens_count) if tokens_count > 1 else 0

        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
        
        word_counts = Counter(words_raw)
        hapax_count = sum(1 for word in word_counts if word_counts[word] == 1)
        
        # Sternen-Daten (Länge + Winkel via Wortart)
        first_sent_doc = nlp(sentences[0]) if sentences else []
        star_data = [{"len": len(t.text), "angle": LoreProcessor.get_pos_angle(t)} for t in first_sent_doc if not t.is_punct]

        # Charakter-Präsenz
        char_presence = {}
        for final_name, alias_list in final_characters.items():
            count = 0
            for alias in alias_list:
                count += len(re.findall(rf'\b{alias}\b', text, re.IGNORECASE))
            char_presence[final_name] = count
        
        return {
            "Sentiment": normalized_sentiment, "Lexical_Density_C": herdan_c,
            "Avg_Sentence_Length": avg_sentence_len, "Hapax": hapax_count,
            "Star_Data": star_data, "Character_Presence": char_presence
        }

# 3. SIDEBAR (Steuerung)
with st.sidebar:
    st.header("📂 Daten-Quelle")
    uploaded_file = st.file_uploader("Lade .txt hoch", type="txt")
    st.divider()
    
    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        st.header("⚙️ Analyse-Basis")
        mode = st.radio("Modus:", ["Kapitel/Szenen", "Wort-Blöcke"])
        
        if mode == "Kapitel/Szenen":
            all_raw_chapters = LoreProcessor.split_into_chapters(raw_text)
            with st.expander("Kapitel filtern", expanded=True):
                selected_indices = []
                for i, c in enumerate(all_raw_chapters):
                    snippet = c[:30].replace('\n', ' ')
                    is_junk = any(x in c.lower()[:300] for x in ["copyright", "gutenberg", "license"])
                    if st.checkbox(f"[{i+1}] {snippet}...", value=not is_junk, key=f"ch_{i}"):
                        selected_indices.append(i)
            chunks = [all_raw_chapters[i] for i in selected_indices]
            labels = [f"Abschnitt {i+1}" for i in selected_indices]
        else:
            block_size = st.slider("Wörter pro Block:", 500, 5000, 1000)
            all_words = raw_text.split()
            chunks = [" ".join(all_words[i : i + block_size]) for i in range(0, len(all_words), block_size)]
            labels = [f"Block {i+1}" for i in range(len(chunks))]

        # 👥 INTERAKTIVES CHARAKTER-MANAGEMENT
        st.header("👥 Charakter-Kopplung")
        if nlp:
            # Schnelle Extraktion für das Interface
            doc_sample = nlp(raw_text[:30000])
            detected = [ent.text.strip() for ent in doc_sample.ents if ent.label_ == "PER" and len(ent.text) > 2]
            top_names = [name for name, count in Counter(detected).most_common(15)]
            
            # Session State für Bündelungen
            if 'char_map' not in st.session_state: st.session_state.char_map = {}
            
            with st.expander("Neue Bündelung erstellen"):
                new_final_name = st.text_input("Name der Hauptfigur (z.B. Luise):")
                aliases = st.multiselect("Zugehörige Varianten wählen:", top_names)
                if st.button("Koppeln"):
                    if new_final_name and aliases:
                        st.session_state.char_map[new_final_name] = aliases
                        st.success(f"{new_final_name} gespeichert.")

            if st.session_state.char_map:
                st.write("**Aktive Bündelungen:**")
                for k, v in st.session_state.char_map.items():
                    st.text(f"{k}: {', '.join(v)}")
                if st.button("Bündelungen zurücksetzen"):
                    st.session_state.char_map = {}
                    st.rerun()

            selected_final_chars = st.multiselect("In Analyse anzeigen:", list(st.session_state.char_map.keys()))
        
        theme_color = st.color_picker("Farbe für Data Art", "#D4AF37")

# 4. HAUPTBEREICH
if uploaded_file and 'chunks' in locals():
    results = []
    star_maps_data = []
    
    progress_bar = st.progress(0, "Analysiere Text...")
    for i, (label, chunk) in enumerate(zip(labels, chunks)):
        m = LoreProcessor.get_metrics(chunk, st.session_state.get('char_map', {}))
        if m:
            row = {"Abschnitt": label, "Sentiment": m["Sentiment"], "Dichte": m["Lexical_Density_C"], 
                   "Satzlänge": m["Avg_Sentence_Length"], "Hapax": m["Hapax"]}
            row.update(m["Character_Presence"])
            results.append(row)
            star_maps_data.append(m["Star_Data"])
        progress_bar.progress((i+1) / len(chunks))
    progress_bar.empty()
    
    df = pd.DataFrame(results)
    tab1, tab2, tab3 = st.tabs(["📊 Narrative Dynamik", "👥 Beziehungs-Netz", "✨ Sternenbild-Poster"])

    with tab1:
        # Sentiment
        st.markdown("#### 📊 Emotionaler Bogen", unsafe_allow_html=True)
        with st.popover("ⓘ Methode & Einordnung"):
            st.write("**Methode:** Normalisierter Abgleich von Valenz-Begriffen pro 1000 Wörter.")
            st.write("**Einordnung:** Werte > 0 zeigen optimistische Passagen, Werte < 0 deuten auf Konflikt oder Tragik hin.")
        fig_sent = px.line(df, x="Abschnitt", y="Sentiment", template="plotly_dark", color_discrete_sequence=[theme_color])
        st.plotly_chart(fig_sent, use_container_width=True)

        st.divider()
        st.markdown("#### 🧠 Lexikalische Dichte (Herdan's C)", unsafe_allow_html=True)
        with st.popover("ⓘ Methode & Einordnung"):
            st.write(r"**Methode:** $C = \log(Types) / \log(Tokens)$. Korrigiert die Längenabhängigkeit.")
            st.write("**Einordnung:** Hohe Werte (>0.8) bedeuten einen sehr diversen, komplexen Wortschatz.")
        fig_dens = px.line(df, x="Abschnitt", y="Dichte", template="plotly_dark", color_discrete_sequence=["#C0C0C0"])
        st.plotly_chart(fig_dens, use_container_width=True)

        st.divider()
        st.markdown("#### ⏱️ Satzlängen-Rhythmus", unsafe_allow_html=True)
        with st.popover("ⓘ Methode & Einordnung"):
            st.write("**Methode:** Durchschnittliche Wörter pro Satz.")
            st.write("**Einordnung:** Kurze Balken = Hektik/Dialoge. Lange Balken = Deskription/Reflexion.")
        fig_len = px.bar(df, x="Abschnitt", y="Satzlänge", template="plotly_dark", color_discrete_sequence=["#444444"])
        st.plotly_chart(fig_len, use_container_width=True)

    with tab2:
        if selected_final_chars:
            st.write("#### 👤 Präsenz der gebündelten Figuren")
            fig_chars = px.line(df, x="Abschnitt", y=selected_final_chars, template="plotly_dark")
            st.plotly_chart(fig_chars, use_container_width=True)
            
            st.divider()
            st.markdown("#### 🤝 Interaktions-Matrix", unsafe_allow_html=True)
            with st.popover("ⓘ Farberklärung"):
                st.write("**Gelb/Hell (1.0):** Die Figuren tauchen fast immer im selben Kapitel auf.")
                st.write("**Dunkel/Violett (0.0):** Die Figuren haben keine gemeinsamen Szenen.")
            corr = df[selected_final_chars].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="Viridis"), use_container_width=True)

    with tab3:
        st.markdown("### 🌌 Nicolas Rougeaux Edition", unsafe_allow_html=True)
        with st.popover("ⓘ Anleitung"):
            st.write("**Urknall:** Alle Wörter des ersten Satzes starten bei Null. Die Wortart (Nomen, Verb etc.) bestimmt die Richtung.")
            st.write("**Sonnensystem:** Jedes Kapitel schwebt autark auf einer Umlaufbahn. Keine Überlagerung.")
        
        variant = st.radio("Poster-Typ:", ["Urknall (Strukturelle Signatur)", "Sonnensystem (Kapitel-Orbit)"])
        fig_stars = go.Figure()
        
        if variant == "Urknall (Strukturelle Signatur)":
            colors = px.colors.sample_colorscale("Viridis", len(star_maps_data))
            for i, data in enumerate(star_maps_data):
                if data:
                    r = [d['len'] for d in data]
                    theta = [d['angle'] for d in data]
                    fig_stars.add_trace(go.Scatterpolar(r=r, theta=theta, mode='markers+lines', name=df['Abschnitt'].iloc[i], line_color=colors[i], marker=dict(size=4)))
            fig_stars.update_layout(polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=True, tickvals=[0, 90, 180, 270], ticktext=["Nomen", "Verb", "Adj", "Adv"])))
        else:
            orbit_angles = np.linspace(0, 360, len(star_maps_data), endpoint=False)
            for i, data in enumerate(star_maps_data):
                if data:
                    # Lokale Rotation + Orbit-Versatz
                    theta = np.linspace(0, 360, len(data), endpoint=False) + orbit_angles[i]
                    fig_stars.add_trace(go.Scatterpolar(r=np.add([d['len'] for d in data], 20), theta=theta, mode='markers+lines', marker_color=theme_color))
            fig_stars.update_layout(polar=dict(radialaxis=dict(visible=False), angularaxis=dict(visible=False)))

        fig_stars.update_layout(height=850, template="plotly_dark")
        st.plotly_chart(fig_stars, use_container_width=True)
