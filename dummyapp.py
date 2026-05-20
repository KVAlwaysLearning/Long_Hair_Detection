import streamlit as st

st.write("Starting imports...")

try:
    import cv2
    st.write("✅ cv2 imported successfully")
except Exception as e:
    st.write(f"❌ cv2 failed: {e}")

try:
    import torch
    st.write("✅ torch imported successfully")
except Exception as e:
    st.write(f"❌ torch failed: {e}")

try:
    import mediapipe as mp
    st.write("✅ mediapipe imported successfully")
except Exception as e:
    st.write(f"❌ mediapipe failed: {e}")

try:
    import transformers
    st.write("✅ transformers imported successfully")
except Exception as e:
    st.write(f"❌ transformers failed: {e}")
