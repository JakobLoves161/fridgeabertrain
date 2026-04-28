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
# SUPABASE INIT
# -----------------------------
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase = create_client(url, key)

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Kühlschrank", layout="centered")

# -----------------------------
# MODELS
# -----------------------------
@st.cache_resource
def load_models():
    import clip
    model, preprocess = clip.load("ViT-B/32")
    return model, preprocess

model, preprocess = load_models()

@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(['de', 'en'])

ocr = load_ocr()

# -----------------------------
# LABELS
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

    "ein Käse","eine Milchpackung","ein Joghurt","ein Quark","ein Frischkäse",
    "ein Stück Butter","eine Sahne","ein Pudding",

    "ein Hähnchen","ein Rindfleisch","ein Schweinefleisch","ein Fischfilet",
    "eine Wurst","ein Schinken","eine Salami",

    "ein Brot","ein Brötchen","eine Pizza","ein Croissant","ein Sandwich",

    "eine Schokolade","ein Keks","eine Packung Chips","ein Eis"
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
# 🍎 FOOD
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
    if manual_food:
        st.session_state.food_item = manual_food

if image:
    st.image(image)

    if st.button("🔍 Erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits, _ = model(img, text_tokens)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        st.session_state.food_item = labels[probs.argmax()]

if st.session_state.food_item:
    st.success(st.session_state.food_item)

# =========================================================
# 📅 MHD
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
    manual_mhd = st.text_input("MHD eingeben")
    if manual_mhd:
        st.session_state.mhd_value = manual_mhd

if mhd_image:
    st.image(mhd_image)

    if st.button("📅 Erkennen"):
        st.session_state.mhd_value = extract_mhd(mhd_image)

# =========================================================
# ➕ SPEICHERN
# =========================================================
st.subheader("➕ Speichern")

if st.session_state.food_item and st.button("In Datenbank speichern"):

    now = datetime.now() + timedelta(hours=2)

    try:
        supabase.table("fridge_inventory").insert({
            "food_name": st.session_state.food_item,
            "mhd": st.session_state.mhd_value,
            "added_at": now.date().isoformat()
        }).execute()

        st.success("Gespeichert!")

        st.session_state.food_item = None
        st.session_state.mhd_value = None

    except Exception as e:
        st.error("❌ Fehler:")
        st.code(str(e))

# =========================================================
# 📦 INVENTAR
# =========================================================
st.subheader("📦 Inventar")

try:
    data = supabase.table("fridge_inventory").select("*").execute().data
except Exception as e:
    st.error("Fehler beim Laden")
    st.code(str(e))
    data = []

if data:

    def parse_date(val):
        try:
            return datetime.fromisoformat(val)
        except:
            return datetime.max

    data = sorted(data, key=lambda x: parse_date(x["mhd"]) if x["mhd"] else datetime.max)

    h1, h2, h3, h4 = st.columns([3,2,2,1])
    h1.markdown("**Lebensmittel**")
    h2.markdown("**MHD**")
    h3.markdown("**Hinzugefügt am**")
    h4.markdown("")

    today = datetime.now().date()

    for row in data:
        c1, c2, c3, c4 = st.columns([3,2,2,1])

        added_date = str(row["added_at"]).split("T")[0]

        color = "white"

        try:
            mhd_date = datetime.fromisoformat(row["mhd"]).date()
            diff = (mhd_date - today).days

            if diff <= 2:
                color = "red"
            elif diff <= 5:
                color = "orange"
        except:
            pass

        c1.markdown(f"<span style='color:{color}'>{row['food_name']}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='color:{color}'>{row['mhd']}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:{color}'>{added_date}</span>", unsafe_allow_html=True)

        if c4.button("❌", key=row["id"]):
            supabase.table("fridge_inventory").delete().eq("id", row["id"]).execute()
            st.rerun()

else:
    st.info("Inventar leer")
