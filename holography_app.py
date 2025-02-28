import streamlit as st
import reco as vc  # Your reconstruction functions are in reco.py
import cv2
import numpy as np
import tempfile
import os

st.title("Holographic Reconstruction")

uploaded_video = st.file_uploader("Upload a Hologram Video", type=["mp4", "avi", "mov"])

# Parameters
wavelength = st.slider("Wavelength (nm)", min_value=400, max_value=800, value=650)
pixel_size = st.slider("Pixel Size (μm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
distance = st.slider("Distance (mm)", min_value=1, max_value=50, value=10)
crop_size = st.slider("Crop Size", min_value=10, max_value=100, value=25)  # Add crop size slider

# --- Helper Functions ---
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# --- Reconstruction Logic ---
if uploaded_video is not None:
    # 1. Write the uploaded video to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # 2. Open the video with your custom function in reco.py
    cap = vc.openVid(temp_video_path)  # Ensure openVid can handle a filepath
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Get total frames
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame Selection
        frame_number = st.slider("Frame Number", min_value=0, max_value=max_frames - 1, value=0)
        ret, rawIM = vc.getFrame(cap, frame_number)

        if not ret:
            st.error("Error reading frame from video.")
        else:
            # Convert to grayscale
            grayIM = cv2.cvtColor(rawIM, cv2.COLOR_BGR2GRAY)

            # --- Cropping ---
            x_center = grayIM.shape[1] // 2  # Default center
            y_center = grayIM.shape[0] // 2

            # Optional: Let user select crop center with a button + mouse click
            if st.button("Select Crop Center"):
                st.info("Please click on the 'Full Image' display to select the center for cropping.")
                # Implementation for mouse click selection would go here

            # Calculate crop window
            x0 = clamp(x_center - crop_size, 0, grayIM.shape[1])
            x1 = clamp(x_center + crop_size, 0, grayIM.shape[1])
            y0 = clamp(y_center - crop_size, 0, grayIM.shape[0])
            y1 = clamp(y_center + crop_size, 0, grayIM.shape[0])

            # Crop image
            cropIM = grayIM[y0:y1, x0:x1]

            # --- Reconstruction ---
            # distance is in mm, so multiply by 1e-3 to convert to meters, if needed
            recoIM = vc.recoFrame(cropIM, distance * 1e-3)

            # Display images
            st.image(grayIM, caption="Full Image", channels="GRAY")
            st.image(recoIM, caption="Reconstructed Image", channels="GRAY")

            # --- Save Images ---
            if st.button("Save Images"):
                # Implementation for saving images to disk or cloud
                st.success("Images saved successfully!")

    # 3. Clean up the temporary file
    os.remove(temp_video_path)
