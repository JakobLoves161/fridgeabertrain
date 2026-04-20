import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime
import clip

# -----------------------------
# Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-B/32")
    return model, preprocess

model, preprocess = load_model()

# -----------------------------
# Labels (50 Lebensmittel)
# -----------------------------
labels = [
    "ein frischer Apfel","eine reife Banane","eine Orange","eine Birne",
    "eine Tomate","eine Gurke","eine Paprika","eine Karotte","eine Kartoffel",
    "eine Zwiebel","eine Knoblauchknolle","ein Brokkoli","ein Blumenkohl",
    "ein Salatkopf","eine Zucchini","eine Aubergine",
    "ein Stück Käse","eine Milchpackung","ein Joghurtbecher","ein Stück Butter",
    "ein Ei","ein rohes Fleischstück","ein Hähnchenfilet","ein Fischfilet",
    "eine Wurst","ein Schinken",
    "eine Brotscheibe","ein ganzes Brot","ein Brötchen","ein Croissant",
    "eine Pizza","ein Sandwich","ein Burger","eine Portion Nudeln","eine Portion Reis",
    "eine Flasche Wasser","eine Saftflasche","eine Cola Flasche",
    "eine Bierflasche","eine Weinflasche",
    "eine Tafel Schokolade","ein Keks","ein Stück Kuchen","ein Eis",
    "ein Joghurt Dessert",
    "eine Dose","eine Konservendose","eine Verpackung Tiefkühlkost","eine Packung Chips"
]

text = clip.tokenize(labels)

# -----------------------------
# Session State
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = []

if "detected_items" not in st.session_state:
    st.session_state.detected_items = []

# -----------------------------
# UI Styling (größerer Upload Bereich)
# -----------------------------
st.markdown("""
    <style>
    .upload-box {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧊 Digitaler Kühlschrank (CLIP AI)")

st.markdown('<div class="upload-box">📸 Lade ein Bild deiner Lebensmittel hoch</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["jpg", "png"])

# -----------------------------
# Bild & Erkennung
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Bild", use_column_width=True)

    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits_per_image, _ = model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # 🔥 Top 5 Ergebnisse
        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]

        detected_items = [labels[i] for i in top_indices]

        st.session_state.detected_items = detected_items

    # Anzeige mehrerer Ergebnisse
    if st.session_state.detected_items:
        st.subheader("🔎 Erkannte Lebensmittel")

        for item in st.session_state.detected_items:
            st.write(f"• {item}")

        if st.button("➕ Alle zum Inventar hinzufügen"):
            for item in st.session_state.detected_items:
                st.session_state.inventory.append({
                    "Lebensmittel": item,
                    "Hinzugefügt am": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

            st.success("✅ Hinzugefügt!")
            st.session_state.detected_items = []

# -----------------------------
# Inventar anzeigen
# -----------------------------
st.subheader("📦 Inventar")

if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)

    for i, row in df.iterrows():
        col1, col2, col3 = st.columns([3, 3, 1])

        col1.write(row["Lebensmittel"])
        col2.write(row["Hinzugefügt am"])

        if col3.button("❌", key=f"delete_{i}"):
            st.session_state.inventory.pop(i)
            st.rerun()
else:
    st.info("Inventar ist leer.")
