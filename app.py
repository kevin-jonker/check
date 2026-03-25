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
    /* Info-I Styling */
    .stPopover { display: inline-block; margin-left: 10px; }
    .stPopover button { border: none; background: none; color: #888; padding: 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("📚 Literary Intelligence System")
st.subheader("Data Science & Narrative DNA")

# 2. LORE-PROCESSOR (Wissenschaftliche Logik)
class LoreProcessor:
    @staticmethod
    def split_into_chapters(text):
        """Verbesserter, robuster Regex für Kapitel-Trenner."""
        pattern = r'\n\s*(?:Kapitel|Chapter|Teil|Buch|SECTION|ACT)\s+[0-9IVXLCDM]+\b|(?:\n\s*[IVXLCDM]+\.\s+\b)'
        chapters = re.split(pattern, text, flags=re.IGNORECASE)
        # Filtert Fragmente unter 150 Zeichen (wie Inhaltsverzeichnisse)
        return [c.strip() for c in chapters if len(c.strip()) > 150]

    @staticmethod
    def get_metrics(text, final_characters):
        words_raw = re.findall(r'\w+', text.lower())
        tokens_count = len(words_raw)
        
        # Falls der Textabschnitt leer ist
        if tokens_count < 10:
            return None
            
        unique_words = set(words_raw)
        types_count = len(unique_words)
        
        # 1. Sentiment (Gewichtet pro 1000 Wörter)
        pos_words = ["liebe", "herz", "freude", "licht", "glück", "lachen", "triumph", "strahlend", "hoffnung", "mut"]
        neg_words = ["tod", "schmerz", "angst", "nacht", "weinen", "elend", "verlust", "grab", "einsam", "zorn"]
        sent_score = sum(1 for w in words_raw if w in pos_words) - sum(1 for w in words_raw if w in neg_words)
        normalized_sentiment = (sent_score / tokens_count * 1000)

        # 2. Herdan's C (Lexikalische Dichte)
        herdan_c = np.log(types_count) / np.log(tokens_count) if tokens_count > 1 else 0

        # 3. Satz-Analyse
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_len = np.mean(sentence_lengths) if sentence_lengths else 0
        sentence_volatility = np.std(sentence_lengths) if sentence_lengths else 0
        
        # 4. Hapax & Sternen-Daten
        word_counts = Counter(words_raw)
        hapax_count = sum(1 for word in word_counts if word_counts[word] == 1)
        first_sentence = sentences[0] if sentences else ""
        star_data = [len(w) for w in first_sentence.split()]

        # 5. Charakter-Präsenz (Bündelung berücksichtigen)
        char_presence = {}
        for final_name, alias_list in final_characters.items():
            count = 0
            for alias in alias_list:
                count += len(re.findall(rf'\b{alias}\b', text, re.IGNORECASE))
            char_presence[final_name] = count
        
        return {
            "Sentiment": normalized_sentiment,
            "Lexical_Density_C": herdan_c,
            "Avg_Sentence_Length": avg_sentence_len,
            "Sentence_Volatility": sentence_volatility,
            "Hapax": hapax_count,
            "First_Sentence": first_sentence,
            "Star_Data": star_data,
            "Character_Presence": char_presence
        }

    @staticmethod
    def get_color_gradient(n, start_color="#D4AF37"):
        """Generiert einen Farbverlauf für die Sternenbilder."""
        colors = px.colors.sample_colorscale("Viridis", n)
        return colors

# 3. SIDEBAR
with st.sidebar:
    st.header("📂 Daten-Upload")
    uploaded_file = st.file_uploader("Lade einen Klassiker (.txt)", type="txt")
    st.divider()
    
    final_characters = {} # Initialisierung

    if uploaded_file:
        raw_text = uploaded_file.read().decode("utf-8")
        st.success("Text geladen!")
        
        # ⚙️ ANALYSE-SETTINGS
        st.header("⚙️ Analyse-Settings")
        mode = st.radio("Analyse-Basis:", ["Wort-Blöcke", "Kapitel-Erkennung"])
        
        if mode == "Kapitel-Erkennung":
            # 🔍 KONTROLL-MECHANISMUS
            with st.expander("Kapitel-Kontrolle", expanded=False):
                temp_chunks = LoreProcessor.split_into_chapters(raw_text)
                st.write(f"**{len(temp_chunks)} Kapitel erkannt.**")
                for i, c in enumerate(temp_chunks):
                    # Zeigt die ersten 50 Zeichen des Kapitels zur Prüfung
                    snippet = c[:50].replace('\n', ' ')
                    st.text(f"[{i+1}] {snippet}...")
        
        if mode == "Wort-Blöcke":
            block_size = st.slider("Wörter pro Block:", 500, 10000, 2000, step=500)
        
        # 👥 CHARAKTER-BÜNDELUNG (Interaktiv & Sauber)
        st.header("👥 Charakter-Management")
        
        # 1. Auto-Erkennung (NLP)
        if nlp:
            with st.spinner("Extrahiere Figuren..."):
                doc_sample = nlp(raw_text[:40000])
                detected = [ent.text.strip() for ent in doc_sample.ents if ent.label_ == "PER" and len(ent.text) > 2]
                top_names = [name for name, count in Counter(detected).most_common(12)]
        else:
            top_names = []
            
        # 2. Bündelungs-Interface
        st.write("Gib Bündelungs-Regeln ein (z.B. `Gregor, Gregors, Samsa -> Gregor`). Eine Regel pro Zeile.")
        bundle_input = st.text_area("Bündelungs-Regeln:", height=150, value="Gregor, Gregors, Samsa -> Gregor")
        
        # Verarbeitung der Regeln
        raw_aliases = Counter(detected)
        rules = bundle_input.split('\n')
        final_characters = {}
        
        # Parse Regeln
        mapped_aliases = set()
        for rule in rules:
            if "->" in rule:
                aliases_part, final_name = rule.split('->')
                final_name = final_name.strip()
                alias_list = [a.strip() for a in aliases_part.split(',') if a.strip()]
                final_characters[final_name] = alias_list
                for a in alias_list: mapped_aliases.add(a)
        
        # Füge verbleibende Top-Namen hinzu, die keine Regeln haben
        for name in top_names:
            if name not in mapped_aliases and name not in final_characters:
                final_characters[name] = [name]
        
        # Auswahl für die Graphen
        all_final_names = list(final_characters.keys())
        selected_final_chars = st.multiselect("Figuren für Analyse wählen:", all_final_names, default=all_final_names[:3])
        
        theme_color = st.color_picker("Akzentfarbe", "#D4AF37")

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
    
    # Filtere Charaktere auf die Auswahl
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
            # Charakter-Counts glatt in die Row schreiben
            for name, count in m["Character_Presence"].items():
                row[name] = count
                
            results.append(row)
            star_maps.append(m["Star_Data"])
        progress_bar.progress((i+1) / len(chunks), f"Verarbeite {label}")
    progress_bar.empty()
    
    df = pd.DataFrame(results)

    # TABS FÜR BESSERE ÜBERSICHT
    tab1, tab2, tab3 = st.tabs(["📊 Narrative Dynamik", "👥 Beziehungs-Netz", "✨ Sternenbild-Poster"])

    with tab1:
        # Layout für Graphen untereinander für bessere Lesbarkeit
        
        # 1. Sentiment
        col1a, col1b = st.columns([4, 1])
        with col1a:
            st.write("#### 📊 Emotionaler Bogen")
            st.write("Der normalisierte Sentiment-Score zeigt die emotionale Ladung des Abschnitts.")
            fig_sent = px.line(df, x="Abschnitt", y="Sentiment", template="plotly_dark", color_discrete_sequence=[theme_color])
            # Achsenbeschriftung und Erklärung
            fig_sent.update_layout(yaxis=dict(title="Score (Normiert pro 1000 Wörter)", zeroline=True, zerolinewidth=2, zerolinecolor='white'), xaxis_title="")
            # Markierung für Positiv/Negativ
            st.plotly_chart(fig_sent, use_container_width=True)
            
        with col1b:
            # INFO POPUP
            with st.popover("ⓘ Info zum Sentiment"):
                st.write("**Methode:** Wir nutzen ein Basis-Lexikon von 20 Begriffen, um positive und negative Wörter zu zählen.")
                st.write("**Skala:**")
                st.success("🟢 **> 0 (Positiv):** Wörter wie 'Liebe', 'Freude', 'Hoffnung' dominieren.")
                st.warning("⚪ **= 0 (Neutral):** Balance der Begriffe oder rein sachlicher Text.")
                st.error("🔴 **< 0 (Negativ):** Wörter wie 'Tod', 'Schmerz', 'Angst' dominieren.")
                st.info("💡 **Erkenntnis:** Wo sind die Wendepunkte im Drama?")

        st.divider()

        # 2. Lexikalische Dichte
        col2a, col2b = st.columns([4, 1])
        with col2a:
            st.write("#### 🧠 Lexikalische Dichte (Herdan's C)")
            fig_dens = px.line(df, x="Abschnitt", y="Dichte_C", template="plotly_dark", color_discrete_sequence=["#C0C0C0"])
            fig_dens.update_layout(yaxis_title="Dichte-Index (Logarithmisch)", xaxis_title="")
            st.plotly_chart(fig_dens, use_container_width=True)
        with col2b:
            with st.popover("ⓘ Info zur Dichte"):
                st.write("**Methode:** Wir nutzen die **Herdan'sche C-Formel** ($C = \frac{\log(Unique)}{\log(Total)}$).")
                st.write("⚪ **Hoher Wert (>0.85):** Komplexe Sprache, seltener Wortschatz, akademisch/klassisch.")
                st.write("⚪ **Niedriger Wert (<0.75):** Einfache Sprache, viele Wiederholungen, umgangssprachlich.")
                st.info("💡 **Erkenntnis:** Verändert der Autor seinen Stil?")

        st.divider()

        # 3. Satzlängen-Rhythmus
        col3a, col3b = st.columns([4, 1])
        with col3a:
            st.write("#### ⏱️ Satzlängen-Rhythmus (Puls)")
            fig_len = px.bar(df, x="Abschnitt", y="Satzlänge", template="plotly_dark", marker_color="#444444")
            fig_len.update_layout(yaxis_title="Ø Wörter pro Satz", xaxis_title="")
            st.plotly_chart(fig_len, use_container_width=True)
        with col3b:
            with st.popover("ⓘ Info zum Rhythmus"):
                st.write("**Methode:** Wir zählen die durchschnittliche Wortanzahl pro Satz.")
                st.write("⚪ **Kurze Sätze (<12 Wörter):** Schnelles Tempo, Action, Hektik, Dialoge.")
                st.write("⚪ **Lange Sätze (>22 Wörter):** Langsames Tempo, Reflexion, Beschreibungen, Kafkaesk.")
                st.info("💡 **Erkenntnis:** Der Herzschlag des Erzählers.")

        st.divider()

        # 4. Vokabular-Reichtum (Hapax)
        col4a, col4b = st.columns([4, 1])
        with col4a:
            st.write("#### ✨ Vokabular-Reichtum (Einzigartige Wörter)")
            fig_hapax = px.line(df, x="Abschnitt", y="Hapax", mode='lines+markers', template="plotly_dark", color_discrete_sequence=["#888888"])
            fig_hapax.update_layout(yaxis_title="Anzahl Hapax Legomena", xaxis_title="")
            st.plotly_chart(fig_hapax, use_container_width=True)
        with col4b:
            with st.popover("ⓘ Info zum Vokabular"):
                st.write("**Methode:** Wir zählen die **Hapax Legomena** – Wörter, die im gesamten Buch nur ein einziges Mal vorkommen.")
                st.write("⚪ **Hoher Wert:** Hohe literarische Qualität, reicher Wortschatz, starke Beschreibungen.")
                st.write("⚪ **Niedriger Wert:** Repetitiver Stil, Fokussierung auf wenige Kernbegriffe.")
                st.info("💡 **Erkenntnis:** Wo sprudelt die sprachliche Kreativität?")

    with tab2:
        if selected_final_chars:
            st.write("#### 👤 Präsenz der Figuren (Zeitverlauf)")
            st.write("Die Linien zeigen, wie oft gebündelte Figuren im Verlauf vorkommen.")
            fig_chars = px.line(df, x="Abschnitt", y=selected_final_chars, template="plotly_dark")
            fig_chars.update_layout(yaxis_title="Anzahl Erwähnungen (Gebündelt)")
            st.plotly_chart(fig_chars, use_container_width=True)
            
            st.divider()
            
            st.write("#### 🤝 Interaktions-Matrix (Wer tritt mit wem auf?)")
            st.write("Diese Matrix zeigt die Korrelation der Namensnennungen.")
            corr = df[selected_final_chars].corr()
            fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis", title="Korrelationsmatrix")
            # Legende für Heatmap
            fig_heat.update_layout(coloraxis_colorbar=dict(title="Korrelation", titleside="top", tickvals=[-1, 0, 1], ticktext=["Unabhängig (-1)", "Neutral (0)", "Gleichzeitig (1)"]))
            
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.write("### 🌌 Nicolas Rougeaux Edition: Erster-Satz-Konstellationen")
        
        # STERNENBILD VARIANTE
        poster_variant = st.radio("Poster-Variante:", ["Urknall (Gemeinsamer Startpunkt mit Farbverlauf)", "Sonnensystem (Verteilte Startpunkte)"])
        
        fig_stars = go.Figure()
        
        num_stars = len(star_maps)
        
        # Generiere Farbverlauf für die "Urknall" Variante (Viridis)
        colors_gradient = LoreProcessor.get_color_gradient(num_stars)

        if poster_variant == "Urknall (Gemeinsamer Startpunkt mit Farbverlauf)":
            for i, star_data in enumerate(star_maps):
                if star_data:
                    # Polare Winkel pro Wort berechnen
                    theta = np.linspace(0, 360, len(star_data), endpoint=False)
                    current_color = colors_gradient[i]
                    
                    fig_stars.add_trace(go.Scatterpolar(
                        r=star_data, theta=theta, mode='markers+lines',
                        name=f"{df['Abschnitt'].iloc[i]}",
                        marker=dict(size=star_data, color=current_color, opacity=0.8),
                        line=dict(color=current_color, width=1)
                    ))
            fig_stars.update_layout(polar=dict(bgcolor="#0e1117", radialaxis=dict(visible=False), angularaxis=dict(visible=False)), title="Der Urknall der Erzählung")

        else: # Sonnensystem (Verteilte Startpunkte)
            # Wir verteilen die Mittelpunkte auf einem Kreis
            theta_chunks = np.linspace(0, 360, num_stars, endpoint=False)
            
            for i, star_data in enumerate(star_maps):
                if star_data:
                    # Berechne Versatz für diesen Chunk
                    offset_theta = theta_chunks[i]
                    theta_base = np.linspace(0, 360, len(star_data), endpoint=False)
                    # Verschiebung im Polaren Koordinatensystem
                    fig_stars.add_trace(go.Scatterpolar(
                        r=np.add(star_data, 10), # Basis-Radius + Wortlänge
                        theta=theta_base, mode='markers+lines',
                        name=f"{df['Abschnitt'].iloc[i]}",
                        marker=dict(size=star_data, color=theme_color, opacity=0.7)
                    ))
            fig_stars.update_layout(polar=dict(bgcolor="#0e1117", radialaxis=dict(visible=False)), title="Sonnensystem der Kapitel")

        fig_stars.update_layout(height=900, template="plotly_dark")
        st.plotly_chart(fig_stars, use_container_width=True)

        # POSTER EXPORT
        if st.button("🖼️ Poster-Export vorbereiten"):
            with st.spinner("Rendere hochauflösendes Poster..."):
                buf = io.BytesIO()
                # Wichtig: Kaleido Engine explizit
                fig_stars.write_image(buf, format="png", scale=2, width=1200, height=1200, engine="kaleido")
                st.session_state.art_poster = buf.getvalue()
                st.success("Poster bereit!")
        
        if "art_poster" in st.session_state:
            st.download_button("📥 PNG-Poster herunterladen", data=st.session_state.art_poster, file_name=f"LiteraryArt_{uploaded_file.name}.png")

else:
    st.info("Bitte lade eine .txt Datei hoch.")
