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
# Labels
# -----------------------------
labels = [
    "ein frischer Apfel",
    "eine reife Banane",
    "eine Orange",
    "eine Birne",
    "eine Tomate",
    "eine Gurke",
    "eine Paprika",
    "eine Karotte",
    "eine Kartoffel",
    "eine Zwiebel",
    "eine Knoblauchknolle",
    "ein Brokkoli",
    "ein Blumenkohl",
    "ein Salatkopf",
    "eine Zucchini",
    "eine Aubergine",

    "ein Stück Käse",
    "eine Milchpackung",
    "ein Joghurtbecher",
    "ein Stück Butter",
    "ein Ei",
    "ein rohes Fleischstück",
    "ein Hähnchenfilet",
    "ein Fischfilet",
    "eine Wurst",
    "ein Schinken",

    "eine Brotscheibe",
    "ein ganzes Brot",
    "ein Brötchen",
    "ein Croissant",
    "eine Pizza",
    "ein Sandwich",
    "ein Burger",
    "eine Portion Nudeln",
    "eine Portion Reis",

    "eine Flasche Wasser",
    "eine Saftflasche",
    "eine Cola Flasche",
    "eine Bierflasche",
    "eine Weinflasche",

    "eine Tafel Schokolade",
    "ein Keks",
    "ein Stück Kuchen",
    "ein Eis",
    "ein Joghurt Dessert",

    "eine Dose",
    "eine Konservendose",
    "eine Verpackung Tiefkühlkost",
    "eine Packung Chips"
]
]

text = clip.tokenize(labels)

# -----------------------------
# Session State
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = []

if "detected_item" not in st.session_state:
    st.session_state.detected_item = None

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Digitaler Kühlschrank (CLIP AI)")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Bild", use_column_width=True)

    # 🔍 ERKENNEN
    if st.button("🔍 Lebensmittel erkennen"):
        img = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits_per_image, _ = model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        predicted_index = probs.argmax()
        detected_item = labels[predicted_index]

        st.session_state.detected_item = detected_item

    # Anzeige des erkannten Items
    if st.session_state.detected_item:
        st.success(f"Erkannt: {st.session_state.detected_item}")

        # ➕ HINZUFÜGEN (jetzt funktioniert!)
        if st.button("➕ Zum Inventar hinzufügen"):
            st.session_state.inventory.append({
                "Lebensmittel": st.session_state.detected_item,
                "Hinzugefügt am": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            st.success("✅ Hinzugefügt!")
            st.session_state.detected_item = None  # reset

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
