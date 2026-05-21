import os
import streamlit as st
import cv2
import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, pipeline
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import gdown

# Set page config for better design
st.set_page_config(layout="wide", page_title="AI Face Analyzer")

# Custom CSS for design and fonts
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    h1 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; text-align: center; }
    .stColumn { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background: white; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def setup_models():
    # 1. DOWNLOAD PHASE
    base_dir = "Models"
    if not os.path.exists(base_dir):
        # Ensure this actually finishes before proceeding
        gdown.download_folder(url="https://drive.google.com/drive/folders/1xLgUm3YgyvgzaL86LD_QFYPTRkHmU_Sd", output=base_dir, quiet=False)

    # 2. PATH DISCOVERY PHASE
    # Define these clearly so there is no ambiguity
    paths = {
        "parsing": os.path.join(base_dir, "hf_models", "face-parsing"),
        "age": os.path.join(base_dir, "hf_models", "age-classifier"),
        "gender": os.path.join(base_dir, "hf_models", "gender-classifier"),
        "landmarker": os.path.join(base_dir, "model_files", "face_landmarker.task"),
        "detector": os.path.join(base_dir, "model_files", "detector.tflite")
    }

    # 3. LOADING PHASE
    # REMOVE model_kwargs entirely. Just pass the path. 
    # Transformers will detect it is a local path and handle it.
    image_processor = SegformerImageProcessor.from_pretrained(paths["parsing"])
    parsing_model = SegformerForSemanticSegmentation.from_pretrained(paths["parsing"])
    
    age_pipe = pipeline("image-classification", model=paths["age"], top_k=None)
    gender_pipe = pipeline("image-classification", model=paths["gender"])

    
    # 4. LOAD MEDIAPIPE (using the base_options with the path)
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=paths["landmarker"]),
            num_faces=1
        )
    )
    
    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=paths["detector"])
        )
    )

    return image_processor, parsing_model, landmarker, age_pipe, gender_pipe, detector

# ==============================================================================
# LOGIC FOR IMAGE ANALYSIS
# ==============================================================================

BUCKET_MAP = {'0-2': 1, '3-9': 6, '10-19': 15, '20-29': 25, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70+': 75}

def run_analysis(image_file, proc, p_model, land, age_p, gen_p, det):
    # --- ADD THIS LINE TO RESET THE POINTER ---
    image_file.seek(0)
    
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w, _ = img.shape

    # Add a check to ensure image is valid
    if img is None:
        st.error("Error: Could not decode image. Please upload a valid image file.")
        return None
        
    # Face Detection
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    det_results = det.detect(mp_img)
    if not det_results.detections: return "No face detected"

    bbox = det_results.detections[0].bounding_box
    face_crop = img[int(bbox.origin_y):int(bbox.origin_y + bbox.height),
                    int(bbox.origin_x):int(bbox.origin_x + bbox.width)]
    pil_face = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    # Age/Gender
    age_res = age_p(pil_face)
    gender_res = gen_p(pil_face)[0]['label']
    best_group = max(age_res, key=lambda x: x['score'])
    exact_age = sum([item['score'] * BUCKET_MAP.get(item['label'], 0) for item in age_res if item['label'] in BUCKET_MAP])

    # Hair Logic
    landmarker_res = land.detect(mp_img)
    chin_y = int(landmarker_res.face_landmarks[0][152].y * h) if landmarker_res.face_landmarks else int(h * 0.65)
    
    inputs = proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt")
    with torch.no_grad(): outputs = p_model(**inputs)
    parsing_map = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False).argmax(dim=1).squeeze(0).numpy()
    hair_mask = cv2.morphologyEx((parsing_map == 13).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    lowest_hair_y = np.max(np.where(np.any(hair_mask > 0, axis=1))[0]) if np.any(hair_mask > 0) else 0
    hair_label = "LONG HAIR" if lowest_hair_y > (chin_y + 20) else "SHORT HAIR / BALD"

    return {
        "Age Group": best_group['label'],
        "Exact Age": round(exact_age, 1),
        "Hair Verdict": hair_label,
        "Final Gender": "male" if (20 <= exact_age <= 30 and "SHORT" in hair_label) else ("female" if (20 <= exact_age <= 30 and "LONG" in hair_label) else gender_res)
    }

# UI Layout
st.title("✨ Long Hair Detection with Age & Conditional Gender")
st.write("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Choose a face image...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

with col2:
    st.subheader("Analysis Results")
    if uploaded_file:
        with st.spinner("Processing deep analysis..."):
            # Initialize
            proc, p_model, land, age_p, gen_p, det = setup_models()
            
            # Run the combined_logic analysis
            # Ensure you pass the uploaded_file object here
            results = run_analysis(uploaded_file, proc, p_model, land, age_p, gen_p, det)
            
            # Displaying Results with nice formatting
            st.metric("Estimated Age", results["Exact Age"])
            st.write(f"**Group:** {results['Age Group']}")
            st.write(f"**Hair Status:** {results['Hair Verdict']}")
            st.write(f"**Gender Inference:** {results['Final Gender']}")
    else:
        st.info("Upload an image to see the AI analysis results here.")
