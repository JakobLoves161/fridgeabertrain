import streamlit as st
import torch
from PIL import Image
import pandas as pd
from datetime import datetime
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
    "eine Traube","eine Zitrone","eine Limette","eine Mango","eine Ananas",
    "eine Tomate","eine Gurke","eine Paprika","eine Karotte","eine Kartoffel",
    "ein Käse","eine Milchpackung","ein Joghurt","ein Brot","eine Pizza",
    "eine Schokolade","ein Keks"
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

# =========================================================
# FOOD ERKENNUNG
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
    manual_food = st.text_input("Lebensmittel eingeben")

    if st.button("Food übernehmen"):
        if manual_food:
            st.session_state.food_item = manual_food

if image:
    st.image(image)

    if st.button("Erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

if st.session_state.food_item:
    st.success(f"🍎 {st.session_state.food_item}")

# =========================================================
# MHD (FULL RESTORED)
# =========================================================
st.subheader("📅 MHD")

mhd_tab1, mhd_tab2, mhd_tab3 = st.tabs(["📷 Kamera", "📁 Upload", "✏️ Manuell"])

mhd_image = None

with mhd_tab1:
    cam_mhd = st.camera_input("MHD Foto")
    if cam_mhd:
        mhd_image = Image.open(cam_mhd)

with mhd_tab2:
    up_mhd = st.file_uploader("MHD Bild", type=["jpg","png"], key="mhd")
    if up_mhd:
        mhd_image = Image.open(up_mhd)

with mhd_tab3:
    manual_mhd = st.text_input("MHD eingeben (z.B. 25.12.2026)")

    if st.button("MHD übernehmen"):
        if manual_mhd:
            st.session_state.mhd_value = manual_mhd

# OCR BUTTON
if mhd_image:
    st.image(mhd_image)

    if st.button("MHD erkennen"):
        st.session_state.mhd_value = extract_mhd(mhd_image)

# =========================================================
# SPEICHERN
# =========================================================
st.subheader("➕ Speichern")

if st.session_state.food_item and st.button("Zum Inventar hinzufügen"):

    today = datetime.now().date().strftime("%Y-%m-%d")  # FIX: nur Datum

    supabase.table("fridge_inventory").insert({
        "food_name": st.session_state.food_item,
        "mhd": st.session_state.mhd_value,
        "added_at": today
    }).execute()

    st.success("Gespeichert!")

    st.session_state.food_item = None
    st.session_state.mhd_value = None

# =========================================================
# INVENTAR + LÖSCHEN + TABELLENÜBERSCHRIFTEN
# =========================================================
st.subheader("📦 Inventar")

data = supabase.table("fridge_inventory").select("*").execute().data

if data:
    df = pd.DataFrame(data)

    df = df.rename(columns={
        "food_name": "Lebensmittel",
        "mhd": "MHD",
        "added_at": "Hinzugefügt am"
    })

    st.dataframe(df, use_container_width=True)

    # ❌ DELETE BUTTONS
    st.write("### ❌ Einzelne Einträge löschen")

    for row in data:
        col1, col2, col3, col4 = st.columns([3,2,2,1])

        col1.write(row["food_name"])
        col2.write(row["mhd"])
        col3.write(row["added_at"])

        if col4.button("❌", key=row["id"]):
            supabase.table("fridge_inventory").delete().eq("id", row["id"]).execute()
            st.rerun()

else:
    st.info("Inventar ist leer")
