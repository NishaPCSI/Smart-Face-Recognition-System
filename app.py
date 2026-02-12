import streamlit as st
import cv2
import numpy as np
import face_recognition
import os

st.set_page_config(page_title="Ultra Fast AI Scanner", layout="wide")
st.title("📸 Smart Face Recognition System")
path = "IMAGES"

@st.cache_resource
def load_known_faces():
    if not os.path.exists(path): os.makedirs(path)
    encodeList, fileNames = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    for file in [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]:
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            encs = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if encs:
                encodeList.append(encs[0])
                fileNames.append(file)
    return encodeList, fileNames

known_encodes, fileNames = load_known_faces()

choice = st.sidebar.radio("Select Input Mode:", ("Live Webcam", "Upload Image"))

if choice == "Live Webcam":
    img_file = st.camera_input("Scan Face")
else:
    img_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png', 'webp'])

if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
    if img is not None:
        img_rgb = cv2.cvtColor(cv2.resize(img, (0,0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        face_encs = face_recognition.face_encodings(img_rgb)
        
        if not face_encs:
            st.error("❌ No face detected.")
        else:
            distances = face_recognition.face_distance(known_encodes, face_encs[0])
            matches = [i for i, d in enumerate(distances) if d < 0.50]

            if matches:
                st.success(f"✅ Total Matches Found: {len(matches)}")
                cols = st.columns(4)
                for i, idx in enumerate(matches):
                    with cols[i % 4]:
                        st.image(os.path.join(path, fileNames[idx]), caption=f"{fileNames[idx]} ({distances[idx]:.2f})")
            else:
                st.error("❌ Unknown Person (No match found)")


