import streamlit as st
import reco as vc  # Your reconstruction functions are in reco.py
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(layout="wide")  # Optional: Use a wide layout
st.title("Holographic Reconstruction")

# === SIDEBAR CONTROLS ===
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a Hologram Video", type=["mp4", "avi", "mov"])
    wavelength = st.slider("Wavelength (nm)", min_value=400, max_value=800, value=650)
    pixel_size = st.slider("Pixel Size (μm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    distance = st.slider("Distance (mm)", min_value=1, max_value=50, value=10)
    crop_size = st.slider("Crop Size", min_value=10, max_value=100, value=25)

# Helper function to clamp values
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# --- MAIN LOGIC ---
if uploaded_video is not None:
    # Write the uploaded video to a temp file so OpenCV can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Open the video using your reco.py function
    cap = vc.openVid(temp_video_path)  # Must accept a filepath
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Get total frames
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Choose which frame to view
        frame_number = st.slider("Frame Number", min_value=0, max_value=max_frames - 1, value=0)
        ret, rawIM = vc.getFrame(cap, frame_number)

        if not ret:
            st.error("Error reading frame from video.")
        else:
            # Convert frame to grayscale
            grayIM = cv2.cvtColor(rawIM, cv2.COLOR_BGR2GRAY)

            # Dimensions
            height, width = grayIM.shape

            # === Sliders to pick X/Y center for cropping ===
            x_center = st.slider("X Center", 0, width - 1, width // 2)
            y_center = st.slider("Y Center", 0, height - 1, height // 2)

            # Draw a crosshair on the original grayscale image (convert to BGR for drawing)
            colorIM = cv2.cvtColor(grayIM, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(
                colorIM,
                (x_center, y_center),
                (0, 255, 0),  # Crosshair color (green)
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2
            )

            # Show the full image with crosshair
            st.image(colorIM, caption="Full Image with Crosshair", channels="BGR")

            # === Crop around the chosen center ===
            x0 = clamp(x_center - crop_size, 0, width)
            x1 = clamp(x_center + crop_size, 0, width)
            y0 = clamp(y_center - crop_size, 0, height)
            y1 = clamp(y_center + crop_size, 0, height)
            cropIM = grayIM[y0:y1, x0:x1]

            # Reconstruct the cropped image
            # (distance is in mm, so multiply by 1e-3 to convert to meters if needed)
            recoIM = vc.recoFrame(cropIM, distance * 1e-3)

            # Show the reconstructed image
            st.image(recoIM, caption="Reconstructed Image", channels="GRAY")

            # Optional: Save Images
            if st.button("Save Images"):
                # (Your logic to save images to disk or elsewhere)
                st.success("Images saved successfully!")

    # Clean up the temp file
    os.remove(temp_video_path)
