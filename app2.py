import streamlit as st
import cv2, faiss, pickle, face_recognition, os, numpy as np

st.set_page_config(page_title="Dynamic AI Scanner", layout="wide")
path = "IMAGES"
if not os.path.exists(path): os.makedirs(path)

def sync_data(index, names):
    current = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    new_files = list(set(current) - set(names))
    if new_files:
        with st.spinner("Syncing new faces..."):
            for f in new_files:
                img = cv2.imread(os.path.join(path, f))
                if img is not None:
                    enc = face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if enc:
                        index.add(np.array([enc[0]]).astype('float32'))
                        names.append(f)
            faiss.write_index(index, "faces.index")
            pickle.dump(names, open("names.pkl", "wb"))
            st.toast("Database Updated!")
    return index, names

@st.cache_resource
def load_data():
    if os.path.exists("faces.index"):
        return faiss.read_index("faces.index"), pickle.load(open("names.pkl", "rb"))
    return faiss.IndexFlatL2(128), []

index, names = sync_data(*load_data())

st.title("📸 Smart Face Recognition System")
st.sidebar.header("Settings")
mode = st.sidebar.radio("Input Mode:", ["Webcam", "Upload Image"])

img_file = st.camera_input("Scan Face") if mode == "Webcam" else st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if img_file:
    img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_encs = face_recognition.face_encodings(rgb)

    if face_encs:
        dist, idxs = index.search(np.array([face_encs[0]]).astype('float32'), k=5)
        cols, found = st.columns(5), False
        for i, idx in enumerate(idxs[0]):
            conf = 1 - dist[0][i]
            if idx != -1 and conf > 0.75:
                found = True
                with cols[i % 5]:
                    st.image(os.path.join(path, names[idx]), caption=f"{conf:.2f}")
                    st.success(names[idx])
        if not found: st.error("No match found with >0.75 confidence.")
    else: st.error("No face detected in the input!")

