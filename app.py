import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from datetime import datetime

# -----------------------------
# YOLO Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # vortrainiertes Modell

model = load_model()

# -----------------------------
# Session State (Inventar)
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = []

# -----------------------------
# UI
# -----------------------------
st.title("🧊 Digitaler Kühlschrank mit YOLOv8")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png"])

# -----------------------------
# Bildverarbeitung + Detection
# -----------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Lebensmittel erkennen"):
        results = model(image)

        detected_items = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                detected_items.append(label)

        if detected_items:
            st.success(f"Erkannt: {', '.join(detected_items)}")

            # Hinzufügen Button
            if st.button("➕ Zum Inventar hinzufügen"):
                for item in detected_items:
                    st.session_state.inventory.append({
                        "Lebensmittel": item,
                        "Hinzugefügt am": datetime.now().strftime("%Y-%m-%d %H:%M")
                    })

                st.success("✅ Zum Inventar hinzugefügt!")
        else:
            st.warning("Keine Lebensmittel erkannt.")

# -----------------------------
# Inventar anzeigen
# -----------------------------
st.subheader("📦 Dein Kühlschrank Inventar")

if st.session_state.inventory:
    df = pd.DataFrame(st.session_state.inventory)

    for i, row in df.iterrows():
        col1, col2, col3 = st.columns([3, 3, 1])

        col1.write(row["Lebensmittel"])
        col2.write(row["Hinzugefügt am"])

        if col3.button("❌", key=i):
            st.session_state.inventory.pop(i)
            st.rerun()

else:
    st.info("Noch keine Lebensmittel im Inventar.")
