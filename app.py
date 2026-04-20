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

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Digitaler Kühlschrank (Auto AI)")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png"])

# -----------------------------
# AUTOMATISCHE ERKENNUNG
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    img = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits_per_image, _ = model(img, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # 🔥 BESTES ERGEBNIS
    best_index = probs.argmax()
    best_label = labels[best_index]
    confidence = probs[best_index]

    # 🔥 Nur hinzufügen wenn sinnvoll sicher
    threshold = 0.20

    if confidence > threshold:
        item = {
            "Lebensmittel": best_label,
            "Hinzugefügt am": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Sicherheit": round(float(confidence), 2)
        }

        # 🔥 Verhindert doppelte Einträge direkt nacheinander
        if not st.session_state.inventory or st.session_state.inventory[-1]["Lebensmittel"] != best_label:
            st.session_state.inventory.append(item)
            st.success(f"✅ Erkannt & hinzugefügt: {best_label} ({confidence:.2f})")
        else:
            st.info("ℹ️ Bereits zuletzt hinzugefügt – kein Duplikat")

    else:
        st.warning("❌ Kein sicheres Lebensmittel erkannt")

# -----------------------------
# INVENTAR
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
