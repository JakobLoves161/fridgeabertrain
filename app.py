import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import clip
import easyocr
import numpy as np
import cv2
import re

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource
def load_models():
    clip_model, preprocess = clip.load("ViT-B/32")
    ocr = easyocr.Reader(['de', 'en'])
    return clip_model, preprocess, ocr

model, preprocess, ocr = load_models()

# -----------------------------
# LABELS (gekürzt hier – du kannst deine 100 behalten)
# -----------------------------
labels = ["ein Apfel","eine Banane","eine Gurke","eine Tomate","ein Käse","ein Brot"]
text = clip.tokenize(labels)

# -----------------------------
# SESSION STATE
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = []

if "food_item" not in st.session_state:
    st.session_state.food_item = None

if "mhd_value" not in st.session_state:
    st.session_state.mhd_value = None

# -----------------------------
# OCR FUNCTION
# -----------------------------
def extract_mhd(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(3.0, (8,8))
    gray = clahe.apply(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    result = ocr.readtext(thresh, detail=0)
    text = " ".join(result)

    match = re.search(r"\d{2}[.\-/]\d{2}[.\-/]\d{2,4}", text)

    return match.group() if match else None

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Smart Kühlschrank KI")

st.info("📸 Tipp: Gute Beleuchtung + nahes Foto für bessere Ergebnisse.")

# -----------------------------
# STEP 1: FOOD (KAMERA)
# -----------------------------
st.subheader("📸 Lebensmittel erkennen")

camera_image = st.camera_input("📷 Foto aufnehmen")

uploaded_image = st.file_uploader("ODER Bild hochladen", type=["jpg","png"])

image = camera_image or uploaded_image

if image:
    image = Image.open(image)
    st.image(image, caption="Eingabe Bild")

    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

    if st.session_state.food_item:
        st.success(f"🍎 Erkannt: {st.session_state.food_item}")

# -----------------------------
# MANUELLE EINGABE (FOOD)
# -----------------------------
st.markdown("### ✏️ Lebensmittel manuell eingeben (Fallback)")

manual_food = st.text_input("z.B. Apfel, Gurke, Käse")

if st.button("📥 Lebensmittel übernehmen"):
    if manual_food:
        st.session_state.food_item = manual_food
        st.success(f"Manuell gesetzt: {manual_food}")

# -----------------------------
# STEP 2: MHD
# -----------------------------
st.subheader("📅 MHD scannen")

mhd_image = st.file_uploader("Bild vom MHD", type=["jpg","png"], key="mhd")

if mhd_image:
    image2 = Image.open(mhd_image)
    st.image(image2, caption="MHD Bild")

    if st.button("📅 MHD erkennen"):
        st.session_state.mhd_value = extract_mhd(image2)

        if st.session_state.mhd_value:
            st.success(f"📅 MHD: {st.session_state.mhd_value}")
        else:
            st.warning("Kein MHD erkannt")

# -----------------------------
# ADD TO INVENTORY
# -----------------------------
st.subheader("➕ Zum Inventar hinzufügen")

if st.session_state.food_item:

    if st.button("Speichern"):
        now = datetime.now() + timedelta(hours=2)

        st.session_state.inventory.append({
            "Lebensmittel": st.session_state.food_item,
            "MHD": st.session_state.mhd_value if st.session_state.mhd_value else "unbekannt",
            "Hinzugefügt": now.strftime("%Y-%m-%d %H:%M")
        })

        st.success("Gespeichert!")

        st.session_state.food_item = None
        st.session_state.mhd_value = None

# -----------------------------
# INVENTAR
# -----------------------------
st.subheader("📦 Inventar")

if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)

    for i, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([3,2,2,1])

        col1.write(row["Lebensmittel"])
        col2.write(row["MHD"])
        col3.write(row["Hinzugefügt"])

        if col4.button("❌", key=i):
            st.session_state.inventory.pop(i)
            st.rerun()
else:
    st.info("Inventar ist leer")
