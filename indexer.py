import face_recognition, cv2, os, faiss, pickle, numpy as np
from tqdm import tqdm

path, encodings, names = "IMAGES", [], []
valid_exts = ('.jpg', '.jpeg', '.png', '.webp')

# 1. Processing Loop (Optimized)
files = [f for f in os.listdir(path) if f.lower().endswith(valid_exts)]
print(f"Total {len(files)} images found.")

for f in tqdm(files):
    img = cv2.imread(os.path.join(path, f))
    if img is not None:
        # Resize aur RGB conversion ek saath
        rgb = cv2.cvtColor(cv2.resize(img, (0,0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encodings.append(enc[0])
            names.append(f)

# 2. FAISS Indexing & Saving
if encodings:
    index = faiss.IndexFlatL2(128)
    index.add(np.array(encodings).astype('float32'))
    
    faiss.write_index(index, "faces.index")
    with open("names.pkl", "wb") as f:
        pickle.dump(names, f)
    print("Success! Index and names saved.")
else:
    print("No faces found.")


























# long formate code
# import face_recognition
# import cv2
# import os
# import numpy as np
# import faiss
# import pickle
# from tqdm import tqdm  # Progress bar ke liye

# # Folder jahan 1 lakh photos hain
# path = "IMAGES"
# encodings = []
# names = []

# # Valid image formats
# valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
# all_files = [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]

# print(f"Total {len(all_files)} images mili hain. Processing shuru ho rahi hai...")

# # Har image ko process karke uska 'vector' nikaalna
# for file in tqdm(all_files):
#     try:
#         img_path = os.path.join(path, file)
#         img = cv2.imread(img_path)
        
#         if img is not None:
#             # Image size chota karna taaki processing fast ho
#             img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
#             rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
#             # Chehre ka mathematical code (Encoding)
#             encs = face_recognition.face_encodings(rgb)
            
#             if encs:
#                 encodings.append(encs[0])
#                 names.append(file)
#     except Exception as e:
#         print(f"Error processing {file}: {e}")

# # --- FAISS INDEXING ---
# if encodings:
#     print("Ab Index file ban rahi hai...")
    
#     # Encodings ko Numpy array mein convert karo
#     encodings_array = np.array(encodings).astype('float32')

#     # FAISS Index: Ye 128 dimensions wala vector space banata hai
#     # 
#     index = faiss.IndexFlatL2(128) 
#     index.add(encodings_array)

#     # 1. Index save karo (Binary file)
#     faiss.write_index(index, "faces.index")
    
#     # 2. Image names save karo (Pickle file)
#     with open("names.pkl", "wb") as f:
#         pickle.dump(names, f)

#     print("Success! 'faces.index' aur 'names.pkl' ban chuki hain.")
# else:
#     print("Koi bhi face detect nahi hua. Folder check karein.")