import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import clip
import numpy as np
import cv2
import re

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource
def load_models():
    clip_model, preprocess = clip.load("ViT-B/32")
    return clip_model, preprocess

model, preprocess = load_models()

# OCR (lazy import optional, falls du easyocr nutzt)
import easyocr
ocr = easyocr.Reader(['de', 'en'])

# -----------------------------
# 🧠 100 FOOD LABELS
# -----------------------------
labels = [
    # OBST
    "ein Apfel","eine Banane","eine Orange","eine Birne","eine Erdbeere",
    "eine Traube","eine Zitrone","eine Limette","eine Mango","eine Ananas",
    "eine Wassermelone","eine Kirsche","ein Pfirsich","eine Nektarine",
    "eine Heidelbeere","eine Himbeere","eine Brombeere","eine Kiwi",
    "eine Granatapfel","eine Grapefruit",

    # GEMÜSE
    "eine Tomate","eine Gurke","eine Paprika","eine Karotte","eine Kartoffel",
    "eine Zwiebel","ein Knoblauch","ein Brokkoli","ein Blumenkohl","ein Salatkopf",
    "eine Zucchini","eine Aubergine","ein Spinat","eine Avocado","ein Pilz",
    "ein Maiskolben","eine Rote Bete","ein Sellerie","eine Lauchzwiebel","ein Kürbis",
    "eine Süßkartoffel","ein Radieschen","eine Erbse","ein Kohlrabi","ein Rosenkohl",

    # MILCHPRODUKTE
    "ein Käse","eine Milchpackung","ein Joghurt","ein Quark","ein Frischkäse",
    "ein Stück Butter","eine Sahne","ein Kefir","ein Pudding","ein Skyr",

    # FLEISCH & FISCH
    "ein Hähnchen","ein Hähnchenfilet","ein Rindfleisch","ein Schweinefleisch","ein Hackfleisch",
    "ein Fischfilet","ein Lachs","eine Forelle","eine Wurst","ein Schinken",
    "eine Salami","ein Schnitzel","eine Bratwurst","ein Steak","ein Thunfisch",

    # BACKWAREN
    "ein Brot","ein Brötchen","ein Baguette","eine Brezel","eine Pizza",
    "ein Croissant","ein Toast","ein Sandwich","ein Donut","ein Muffin",

    # FERTIGGERICHTE
    "eine Tiefkühlpizza","eine Lasagne","eine Suppe","eine Nudelschale",
    "eine Reisportion","ein Burger","ein Curry","eine Fertigmahlzeit",

    # GETRÄNKE
    "eine Wasserflasche","eine Saftflasche","eine Cola","eine Limonade",
    "eine Bierflasche","eine Weinflasche","eine Milch",

    # SNACKS
    "eine Schokolade","ein Keks","ein Riegel","eine Packung Chips","ein Eis"
]

text_tokens = clip.tokenize(labels)

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

st.info("📸 Tipp: Gute Beleuchtung & nahes Foto verbessern die Erkennung stark.")

# =========================================================
# 🍎 FOOD ERKENNUNG
# =========================================================
st.subheader("📸 Lebensmittel erkennen")

food_tab1, food_tab2, food_tab3 = st.tabs(["📷 Kamera", "📁 Upload", "✏️ Manuell"])

image = None

with food_tab1:
    cam = st.camera_input("Foto aufnehmen")
    if cam:
        image = Image.open(cam)

with food_tab2:
    up = st.file_uploader("Bild hochladen", type=["jpg","png"])
    if up:
        image = Image.open(up)

with food_tab3:
    manual_food = st.text_input("Lebensmittel eingeben")
    if st.button("Food übernehmen"):
        if manual_food:
            st.session_state.food_item = manual_food
            st.success(f"Manuell gesetzt: {manual_food}")

if image:
    st.image(image, caption="Food Bild")

    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

    if st.session_state.food_item:
        st.success(f"🍎 Erkannt: {st.session_state.food_item}")

# =========================================================
# 📅 MHD ERKENNUNG
# =========================================================
st.subheader("📅 MHD erkennen")

mhd_tab1, mhd_tab2 = st.tabs(["📷 Kamera", "📁 Upload"])

mhd_image = None

with mhd_tab1:
    cam_mhd = st.camera_input("MHD Foto")
    if cam_mhd:
        mhd_image = Image.open(cam_mhd)

with mhd_tab2:
    up_mhd = st.file_uploader("MHD Bild", type=["jpg","png"], key="mhd")
    if up_mhd:
        mhd_image = Image.open(up_mhd)

if mhd_image:
    st.image(mhd_image, caption="MHD Bild")

    if st.button("📅 MHD erkennen"):
        st.session_state.mhd_value = extract_mhd(mhd_image)

        if st.session_state.mhd_value:
            st.success(f"📅 MHD: {st.session_state.mhd_value}")
        else:
            st.warning("Kein MHD erkannt")

# -----------------------------
# ✏️ MANUELLES MHD
# -----------------------------
st.markdown("### ✏️ MHD manuell eingeben")

manual_mhd = st.text_input("z.B. 25.12.2026")

if st.button("MHD übernehmen"):
    if manual_mhd:
        st.session_state.mhd_value = manual_mhd
        st.success(f"Manuell gesetzt: {manual_mhd}")

# =========================================================
# ➕ INVENTAR
# =========================================================
st.subheader("➕ Inventar speichern")

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

# =========================================================
# 📦 INVENTAR
# =========================================================
st.subheader("📦 Inventar")

if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)

    for i, row in df.iterrows():
        c1, c2, c3, c4 = st.columns([3,2,2,1])

        c1.write(row["Lebensmittel"])
        c2.write(row["MHD"])
        c3.write(row["Hinzugefügt"])

        if c4.button("❌", key=i):
            st.session_state.inventory.pop(i)
            st.rerun()
else:
    st.info("Inventar ist leer")
