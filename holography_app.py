import streamlit as st
import reco as vc  # Your reconstruction functions are in reco.py
import cv2
import numpy as np
import tempfile
import os

# Make the page layout "wide" so there's more horizontal space
st.set_page_config(layout="wide")

# Title at the top
st.title("Holographic Reconstruction")

# === SIDEBAR CONTROLS ===
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a Hologram Video", type=["mp4", "avi", "mov"])

    # Sliders
    wavelength = st.slider("Wavelength (nm)", min_value=400, max_value=800, value=650)
    pixel_size = st.slider("Pixel Size (μm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    distance = st.slider("Distance (mm)", min_value=1, max_value=50, value=10)
    crop_size = st.slider("Crop Size", min_value=10, max_value=100, value=25)

    # A button or any other controls can also go here
    st.write("---")
    st.write("Adjust parameters in the sidebar to reconstruct your hologram.")

# Helper function to clamp values for cropping
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# === MAIN LOGIC ===
if uploaded_video is not None:
    # Write the uploaded video to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Open the video with your custom function
    cap = vc.openVid(temp_video_path)  # Ensure vc.openVid expects a filepath
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Get total frames
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Let user pick a frame
        frame_number = st.slider("Frame Number", min_value=0, max_value=max_frames - 1, value=0)
        ret, rawIM = vc.getFrame(cap, frame_number)

        if not ret:
            st.error("Error reading frame from video.")
        else:
            # Convert to grayscale
            grayIM = cv2.cvtColor(rawIM, cv2.COLOR_BGR2GRAY)

            # Default crop center (middle of the image)
            x_center = grayIM.shape[1] // 2
            y_center = grayIM.shape[0] // 2

            # Crop window
            x0 = clamp(x_center - crop_size, 0, grayIM.shape[1])
            x1 = clamp(x_center + crop_size, 0, grayIM.shape[1])
            y0 = clamp(y_center - crop_size, 0, grayIM.shape[0])
            y1 = clamp(y_center + crop_size, 0, grayIM.shape[0])
            cropIM = grayIM[y0:y1, x0:x1]

            # Reconstruct (distance is in mm, so multiply by 1e-3 if needed)
            recoIM = vc.recoFrame(cropIM, distance * 1e-3)

            # === LAYOUT FOR IMAGES SIDE BY SIDE (CENTERED) ===
            # We create four columns: empty "spacers" on the left and right, and two main columns in the middle.
            col_left, col_img1, col_img2, col_right = st.columns([1, 3, 3, 1])

            with col_img1:
                st.image(grayIM, caption="Full Image", channels="GRAY", use_container_width=True)

            with col_img2:
                st.image(recoIM, caption="Reconstructed Image", channels="GRAY", use_container_width=True)

            # Optional: Save button
            if st.button("Save Images"):
                # (Your saving logic here)
                st.success("Images saved successfully!")

    # Remove temp file
    os.remove(temp_video_path)
