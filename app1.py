import streamlit as st
import os
import gdown
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, pipeline
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
DRIVE_FOLDER_ID = "1xLgUm3YgyvgzaL86LD_QFYPTRkHmU_Sd"
ROOT_DIR = os.getcwd()
MODELS_ROOT = os.path.join(ROOT_DIR, "Models")

@st.cache_resource
def setup_environment():
    # 1. Ensure download is done
    if not os.path.exists(MODELS_ROOT):
        with st.spinner("Downloading models..."):
            gdown.download_folder(id=DRIVE_FOLDER_ID, output=MODELS_ROOT, quiet=False)
            time.sleep(5)

    # 2. DYNAMIC PATH DISCOVERY
    # This finds the actual path where hf_models is located, 
    # regardless of how many nested folders gdown created.
    hf_root = None
    model_files_dir = None
    
    for root, dirs, files in os.walk(MODELS_ROOT):
        if "hf_models" in dirs and hf_root is None:
            hf_root = os.path.join(root, "hf_models")
        if "model_files" in dirs and model_files_dir is None:
            model_files_dir = os.path.join(root, "model_files")
            
    if not hf_root or not model_files_dir:
        st.error(f"Could not find model subfolders. Found: {list(os.walk(MODELS_ROOT))}")
        return None

    # 3. Load Models using the found paths
    parsing_path = os.path.join(hf_root, "face-parsing")
    
    proc = SegformerImageProcessor.from_pretrained(parsing_path, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(parsing_path, local_files_only=True)

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=os.path.join(model_files_dir, "face_landmarker.task")), num_faces=1)
    )
    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(base_options=python.BaseOptions(model_asset_path=os.path.join(model_files_dir, "detector.tflite")))
    )
    
    age_pipe = pipeline("image-classification", model=os.path.join(hf_root, "age-classifier"), local_files_only=True)
    gender_pipe = pipeline("image-classification", model=os.path.join(hf_root, "gender-classifier"), local_files_only=True)
    
    return proc, model, landmarker, detector, age_pipe, gender_pipe

# --- APP LOGIC ---
env_data = setup_environment()
if not env_data:
    st.error("Model loading failed.")
    st.stop()

proc, model, landmarker, detector, age_pipe, gender_pipe = env_data
BUCKET_MAP = {'0-2': 1, '3-9': 6, '10-19': 15, '20-29': 25, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70+': 75}

def process_image(image_file):
    bytes_data = image_file.read()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)
    h, w, _ = img.shape
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    results = detector.detect(mp_img)
    if not results.detections: return None, "No face detected"
    
    det = results.detections[0].bounding_box
    face_crop = img[int(det.origin_y):int(det.origin_y + det.height), int(det.origin_x):int(det.origin_x + det.width)]
    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    
    age_res = age_pipe(pil_face)
    gender_res = gender_pipe(pil_face)[0]['label']
    exact_age = sum([item['score'] * BUCKET_MAP.get(item['label'], 0) for item in age_res if item['label'] in BUCKET_MAP])

    land_res = landmarker.detect(mp_img)
    chin_y = int(land_res.face_landmarks[0][152].y * h) if land_res.face_landmarks else int(h * 0.65)
    
    inputs = proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt")
    with torch.no_grad(): outputs = model(**inputs)
    parsing_map = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False).argmax(dim=1).squeeze(0).cpu().numpy()
    
    hair_mask = cv2.morphologyEx((parsing_map == 13).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    lowest_hair_y = np.max(np.where(np.any(hair_mask > 0, axis=1))[0]) if np.any(hair_mask > 0) else 0
    
    hair_label = "LONG HAIR" if lowest_hair_y > (chin_y + 20) else "SHORT HAIR / BALD"
    return {"Age": round(exact_age, 1), "Hair": hair_label, "Gender": gender_res}, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- UI ---
st.title("Face Analysis Engine")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
if uploaded_file:
    res, img_or_err = process_image(uploaded_file)
    if isinstance(res, dict):
        st.image(img_or_err)
        st.write(res)
    else:
        st.error(img_or_err)
