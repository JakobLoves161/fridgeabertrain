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

if "food_item" not in st.session_state:
    st.session_state.food_item = None

if "mhd_value" not in st.session_state:
    st.session_state.mhd_value = None

# -----------------------------
# 🔥 VERBESSERTE OCR FUNCTION
# -----------------------------
def extract_mhd(image):
    img = np.array(image)

    for angle in [0, 90, 180, 270]:
        rotated = np.rot90(img, k=angle // 90)

        # Graustufen
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

        # Kontrast verbessern (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Schärfen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp = cv2.filter2D(gray, -1, kernel)

        # Threshold
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR
        result = ocr.readtext(thresh, detail=0)
        text = " ".join(result)

        # 🔥 robuste Datums-Erkennung
        match = re.search(
            r"(\d{2}[.\-/]\d{2}[.\-/]\d{2,4})|"
            r"(\d{4}[.\-/]\d{2}[.\-/]\d{2})|"
            r"(\d{2}[.\-/]\d{2})",
            text
        )

        if match:
            return match.group()

    return None

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Smart Kühlschrank KI")

st.info("📸 Tipp: Mache ein scharfes, nahes Foto vom Mindesthaltbarkeitsdatum (gut beleuchtet, gerade).")

# -----------------------------
# STEP 1: FOOD
# -----------------------------
st.subheader("📸 Schritt 1: Lebensmittel erkennen")

food_image = st.file_uploader("Bild vom Lebensmittel", type=["jpg", "png"], key="food")

if food_image:
    image1 = Image.open(food_image)
    st.image(image1, caption="Lebensmittel")

    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image1).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        best_index = probs.argmax()
        st.session_state.food_item = labels[best_index]

    if st.session_state.food_item:
        st.success(f"🍎 Erkannt: {st.session_state.food_item}")

# -----------------------------
# STEP 2: MHD
# -----------------------------
st.subheader("📅 Schritt 2: MHD scannen")

mhd_image = st.file_uploader("Bild vom MHD", type=["jpg", "png"], key="mhd")

if mhd_image:
    image2 = Image.open(mhd_image)
    st.image(image2, caption="MHD Bild")

    if st.button("📅 MHD erkennen"):
        mhd = extract_mhd(image2)

        if mhd:
            st.session_state.mhd_value = mhd
            st.success(f"📅 MHD erkannt: {mhd}")
        else:
            st.warning("❌ Kein MHD gefunden – bitte näher fotografieren")

if st.session_state.mhd_value:
    st.info(f"📅 Gespeichertes MHD: {st.session_state.mhd_value}")

# -----------------------------
# STEP 3: ADD
# -----------------------------
st.subheader("➕ Zum Inventar hinzufügen")

if st.session_state.food_item:

    if st.button("➕ Speichern"):
        now = datetime.now() + timedelta(hours=2)

        st.session_state.inventory.append({
            "Lebensmittel": st.session_state.food_item,
            "MHD": st.session_state.mhd_value if st.session_state.mhd_value else "unbekannt",
            "Hinzugefügt": now.strftime("%Y-%m-%d %H:%M")
        })

        st.success("✅ Erfolgreich gespeichert!")

        # reset
        st.session_state.food_item = None
        st.session_state.mhd_value = None

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
