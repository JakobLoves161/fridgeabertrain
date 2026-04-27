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
# 🧠 100 LABELS
# -----------------------------
labels = [
    "ein Apfel","eine Banane","eine Orange","eine Birne","eine Erdbeere",
    "eine Traube","eine Zitrone","eine Limette","eine Mango","eine Ananas",
    "eine Wassermelone","eine Kirsche","ein Pfirsich","eine Nektarine",
    "eine Heidelbeere","eine Himbeere","eine Brombeere","eine Kiwi",
    "eine Granatapfel","eine Grapefruit",

    "eine Tomate","eine Gurke","eine Paprika","eine Karotte","eine Kartoffel",
    "eine Zwiebel","ein Knoblauch","ein Brokkoli","ein Blumenkohl","ein Salatkopf",
    "eine Zucchini","eine Aubergine","ein Spinat","eine Avocado","ein Pilz",
    "ein Maiskolben","eine Rote Bete","ein Sellerie","eine Lauchzwiebel","ein Kürbis",
    "eine Süßkartoffel","ein Radieschen","eine Erbse","ein Kohlrabi","ein Rosenkohl",

    "ein Käse","eine Milchpackung","ein Joghurt","ein Quark","ein Frischkäse",
    "ein Stück Butter","eine Sahne","ein Pudding",

    "ein Hähnchen","ein Hähnchenfilet","ein Rindfleisch","ein Schweinefleisch","ein Hackfleisch",
    "ein Fischfilet","ein Lachs","eine Forelle","eine Wurst","ein Schinken",
    "eine Salami","ein Schnitzel","eine Bratwurst","ein Steak","ein Thunfisch",

    "ein Brot","ein Brötchen","ein Baguette","eine Brezel","eine Pizza",
    "ein Croissant","ein Toast","ein Sandwich","ein Donut","ein Muffin",

    "eine Tiefkühlpizza","eine Lasagne","eine Suppe","eine Nudelschale",
    "eine Reisportion","ein Burger","ein Curry","eine Fertigmahlzeit",

    "eine Wasserflasche","eine Saftflasche","eine Cola","eine Limonade",
    "eine Bierflasche","eine Weinflasche","eine Milch",

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
    up = st.file_uploader("Bild hochladen", type=["jpg", "png"])
    if up:
        image = Image.open(up)

with food_tab3:
    manual_food = st.text_input("Lebensmittel eingeben")

    if st.button("Übernehmen Food"):
        if manual_food:
            st.session_state.food_item = manual_food

if image:
    st.image(image)

    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

if st.session_state.food_item:
    st.success(f"🍎 {st.session_state.food_item}")

# =========================================================
# 📅 MHD (KAMERA / UPLOAD / MANUELL)
# =========================================================
st.subheader("📅 MHD erkennen")

mhd_tab1, mhd_tab2, mhd_tab3 = st.tabs(["📷 Kamera", "📁 Upload", "✏️ Manuell"])

mhd_image = None

with mhd_tab1:
    cam_mhd = st.camera_input("MHD Foto")
    if cam_mhd:
        mhd_image = Image.open(cam_mhd)

with mhd_tab2:
    up_mhd = st.file_uploader("MHD Bild", type=["jpg", "png"], key="mhd")
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

    if st.button("📅 MHD erkennen"):
        st.session_state.mhd_value = extract_mhd(mhd_image)

# =========================================================
# ➕ SPEICHERN IN SUPABASE
# =========================================================
st.subheader("➕ Inventar speichern")

if st.session_state.food_item:

    if st.button("Zum Inventar hinzufügen"):

        today = datetime.now().date()   # 👉 NUR DATUM

        supabase.table("fridge_inventory").insert({
            "food_name": st.session_state.food_item,
            "mhd": st.session_state.mhd_value,
            "added_at": str(today)
        }).execute()

        st.success("Gespeichert!")

        st.session_state.food_item = None
        st.session_state.mhd_value = None

# =========================================================
# 📦 INVENTAR + LÖSCHEN
# =========================================================
st.subheader("📦 Inventar")

data = supabase.table("fridge_inventory").select("*").execute().data

if data:
    for row in data:
        c1, c2, c3, c4 = st.columns([3,2,2,1])

        c1.write(row["food_name"])
        c2.write(row["mhd"])
        c3.write(row["added_at"])

        if c4.button("❌", key=row["id"]):
            supabase.table("fridge_inventory").delete().eq("id", row["id"]).execute()
            st.rerun()
else:
    st.info("Inventar ist leer")
