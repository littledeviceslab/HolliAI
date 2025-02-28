import streamlit as st
import reco as vc  # Your reconstruction functions from reco.py
import cv2
import numpy as np
import tempfile
import os

# Set the page to wide mode for more horizontal space
st.set_page_config(layout="wide")
st.title("Holographic Reconstruction")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a Hologram Video", type=["mp4", "avi", "mov"])
    wavelength = st.slider("Wavelength (nm)", min_value=400, max_value=800, value=650)
    pixel_size = st.slider("Pixel Size (μm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    distance = st.slider("Distance (mm)", min_value=1, max_value=250, value=10)
    crop_size = st.slider("Crop Size", min_value=10, max_value=100, value=25)

# Helper function for clamping values
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# --- MAIN LOGIC ---
if uploaded_video is not None:
    # Write the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Open the video using your custom function from reco.py
    cap = vc.openVid(temp_video_path)  # Ensure this accepts a file path
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Select a frame to work with
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = st.slider("Frame Number", min_value=0, max_value=max_frames - 1, value=0)
        ret, rawIM = vc.getFrame(cap, frame_number)
        if not ret:
            st.error("Error reading frame from video.")
        else:
            # Convert the frame to grayscale
            grayIM = cv2.cvtColor(rawIM, cv2.COLOR_BGR2GRAY)
            height, width = grayIM.shape

            # --- X/Y Center Sliders ---
            x_center = st.slider("X Center", 0, width - 1, width // 2)
            y_center = st.slider("Y Center", 0, height - 1, height // 2)

            # --- Draw Crosshair on the Original Image ---
            colorIM = cv2.cvtColor(grayIM, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(
                colorIM,
                (x_center, y_center),
                (0, 255, 0),  # Green color
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2
            )

            # --- Calculate Crop Window ---
            x0 = clamp(x_center - crop_size, 0, width)
            x1 = clamp(x_center + crop_size, 0, width)
            y0 = clamp(y_center - crop_size, 0, height)
            y1 = clamp(y_center + crop_size, 0, height)
            cropIM = grayIM[y0:y1, x0:x1]

            # --- Reconstruction ---
            # Convert distance from mm to meters if needed
            recoIM = vc.recoFrame(cropIM, distance * 1e-3)

            # Resize the reconstructed image to match the original image size
            recoIM_resized = cv2.resize(recoIM, (width, height), interpolation=cv2.INTER_LINEAR)

            # --- Layout: Original on Left, Reconstructed on Right ---
            col1, col2 = st.columns(2)
            with col1:
                st.image(colorIM, caption="Original Image with Crosshair", channels="BGR", width=600)
            with col2:
                st.image(recoIM_resized, caption="Reconstructed Image (Resized)", channels="GRAY", width=600)

            # Optional: Save Images
            if st.button("Save Images"):
                # (Your save logic here)
                st.success("Images saved successfully!")

    # Clean up the temporary file
    os.remove(temp_video_path)
