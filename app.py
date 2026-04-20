import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import clip
import easyocr
import re

# -----------------------------
# MODELLS
# -----------------------------
@st.cache_resource
def load_models():
    clip_model, preprocess = clip.load("ViT-B/32")
    ocr = easyocr.Reader(['de', 'en'])
    return clip_model, preprocess, ocr

model, preprocess, ocr = load_models()

# -----------------------------
# LABELS
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
# SESSION STATE
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = []

if "detected_item" not in st.session_state:
    st.session_state.detected_item = None

if "mhd" not in st.session_state:
    st.session_state.mhd = None

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_mhd(image):
    """OCR + Datumserkennung"""
    result = ocr.readtext(image)

    text = " ".join([r[1] for r in result])

    # typische Datumsformate
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})|(\d{4}-\d{2}-\d{2})", text)

    if match:
        return match.group()
    return None

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Digitaler Kühlschrank (CLIP + MHD Scanner)")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

# -----------------------------
# ERKENNUNG
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(img_tensor, text)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    best_index = probs.argmax()
    st.session_state.detected_item = labels[best_index]

    st.success(f"Erkannt: {st.session_state.detected_item}")

    # -----------------------------
    # MHD SCAN
    # -----------------------------
    if st.button("📅 MHD scannen"):
        mhd = extract_mhd(uploaded_file)

        if mhd:
            st.session_state.mhd = mhd
            st.success(f"MHD erkannt: {mhd}")
        else:
            st.warning("Kein MHD gefunden")

    # Anzeige
    if st.session_state.mhd:
        st.info(f"📅 MHD: {st.session_state.mhd}")

    # -----------------------------
    # HINZUFÜGEN BUTTON
    # -----------------------------
    if st.button("➕ Zum Inventar hinzufügen"):
        now = datetime.now() + timedelta(hours=2)  # 🔥 +2 Stunden

        st.session_state.inventory.append({
            "Lebensmittel": st.session_state.detected_item,
            "MHD": st.session_state.mhd if st.session_state.mhd else "unbekannt",
            "Hinzugefügt": now.strftime("%Y-%m-%d %H:%M")
        })

        st.success("✅ Hinzugefügt!")

        # reset
        st.session_state.detected_item = None
        st.session_state.mhd = None

# -----------------------------
# INVENTAR
# -----------------------------
st.subheader("📦 Inventar")

if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)

    for i, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        col1.write(row["Lebensmittel"])
        col2.write(f"MHD: {row['MHD']}")
        col3.write(row["Hinzugefügt"])

        if col4.button("❌", key=f"del_{i}"):
            st.session_state.inventory.pop(i)
            st.rerun()
else:
    st.info("Inventar ist leer")
