import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import clip
import numpy as np
import cv2
import re
import easyocr
from supabase import create_client

# -----------------------------
# SUPABASE
# -----------------------------
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource
def load_models():
    clip_model, preprocess = clip.load("ViT-B/32")
    return clip_model, preprocess

model, preprocess = load_models()
ocr = easyocr.Reader(['de', 'en'])

# -----------------------------
# LABELS
# -----------------------------
labels = [
    "ein Apfel","eine Banane","eine Orange","eine Birne","eine Erdbeere",
    "eine Tomate","eine Gurke","ein Käse","eine Milchpackung","ein Brot",
    "eine Pizza","ein Joghurt","eine Schokolade"
]

text_tokens = clip.tokenize(labels)

# -----------------------------
# SESSION STATE
# -----------------------------
if "food_item" not in st.session_state:
    st.session_state.food_item = None

if "mhd_value" not in st.session_state:
    st.session_state.mhd_value = None

# -----------------------------
# OCR
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

# =========================================================
# FOOD
# =========================================================
st.subheader("📸 Lebensmittel")

food_tab1, food_tab2, food_tab3 = st.tabs(["📷 Kamera", "📁 Upload", "✏️ Manuell"])

image = None

with food_tab1:
    cam = st.camera_input("Foto")
    if cam:
        image = Image.open(cam)

with food_tab2:
    up = st.file_uploader("Upload", type=["jpg","png"])
    if up:
        image = Image.open(up)

with food_tab3:
    manual = st.text_input("Food eingeben")
    if st.button("Übernehmen"):
        if manual:
            st.session_state.food_item = manual

if image:
    st.image(image)

    if st.button("Erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

if st.session_state.food_item:
    st.success(st.session_state.food_item)

# =========================================================
# MHD
# =========================================================
st.subheader("📅 MHD")

mhd_image = st.file_uploader("MHD Bild", type=["jpg","png"], key="mhd")

if mhd_image:
    image = Image.open(mhd_image)
    st.image(image)

    if st.button("MHD erkennen"):
        st.session_state.mhd_value = extract_mhd(image)

# =========================================================
# SAVE TO SUPABASE
# =========================================================
st.subheader("➕ Speichern")

if st.session_state.food_item and st.button("Speichern"):

    now = datetime.now() + timedelta(hours=2)

    supabase.table("fridge_inventory").insert({
        "food_name": st.session_state.food_item,
        "mhd": st.session_state.mhd_value,
        "added_at": now.isoformat()
    }).execute()

    st.success("Gespeichert!")

    st.session_state.food_item = None
    st.session_state.mhd_value = None

# =========================================================
# LOAD DATA
# =========================================================
st.subheader("📦 Inventar")

data = supabase.table("fridge_inventory").select("*").execute().data

for row in data:
    c1, c2, c3 = st.columns([3,2,2])

    c1.write(row["food_name"])
    c2.write(row["mhd"])
    c3.write(row["added_at"])
