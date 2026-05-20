import streamlit as st
import gdown
import os

st.title("Gdown Download Test")
                    
DRIVE_FOLDER_ID = "1xLgUm3YgyvgzaL86LD_QFYPTRkHmU_Sd"
TARGET_DIR = os.path.join(os.getcwd(), "Models")

if st.button("Start Download"):
    st.write(f"Downloading to: {TARGET_DIR}")
    try:
        # Download the folder
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=TARGET_DIR, quiet=False)
        st.success("Download complete!")
        
        # Walk through the directory to see what was created
        st.write("--- Directory Structure ---")
        for root, dirs, files in os.walk(TARGET_DIR):
            level = root.replace(TARGET_DIR, '').count(os.sep)
            indent = ' ' * 4 * (level)
            st.write(f"{indent}{os.path.basename(root)}/")
            for f in files:
                st.write(f"{indent}    {f}")
                
    except Exception as e:
        st.error(f"Download failed: {e}")
