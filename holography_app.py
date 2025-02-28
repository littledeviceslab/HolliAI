import streamlit as st
import reco as vc  # Your reconstruction functions are in reco.py
import cv2
import numpy as np
import tempfile
import os

# For optional interactive clicking:
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Set the page to a wide layout so we have more horizontal space
st.set_page_config(layout="wide")

# Title
st.title("Holographic Reconstruction")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    uploaded_video = st.file_uploader("Upload a Hologram Video", type=["mp4", "avi", "mov"])
    wavelength = st.slider("Wavelength (nm)", min_value=400, max_value=800, value=650)
    pixel_size = st.slider("Pixel Size (μm)", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
    distance = st.slider("Distance (mm)", min_value=1, max_value=50, value=10)
    crop_size = st.slider("Crop Size", min_value=10, max_value=100, value=25)
    st.write("---")
    st.write("Adjust parameters in the sidebar.")

# Helper function to clamp values for cropping
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# Session state for storing crop center and selection mode
if "crop_center" not in st.session_state:
    st.session_state["crop_center"] = None
if "select_mode" not in st.session_state:
    st.session_state["select_mode"] = False

# --- MAIN LOGIC ---
if uploaded_video is not None:
    # 1) Write the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # 2) Open video with your custom function
    cap = vc.openVid(temp_video_path)  # vc.openVid must accept a filepath
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = st.slider("Frame Number", min_value=0, max_value=max_frames - 1, value=0)
        ret, rawIM = vc.getFrame(cap, frame_number)

        if not ret:
            st.error("Error reading frame from video.")
        else:
            # Convert to grayscale
            grayIM = cv2.cvtColor(rawIM, cv2.COLOR_BGR2GRAY)

            # Default center is the image center, unless the user picked a custom one
            if st.session_state["crop_center"] is None:
                x_center = grayIM.shape[1] // 2
                y_center = grayIM.shape[0] // 2
            else:
                x_center, y_center = st.session_state["crop_center"]

            # Crop window
            x0 = clamp(x_center - crop_size, 0, grayIM.shape[1])
            x1 = clamp(x_center + crop_size, 0, grayIM.shape[1])
            y0 = clamp(y_center - crop_size, 0, grayIM.shape[0])
            y1 = clamp(y_center + crop_size, 0, grayIM.shape[0])
            cropIM = grayIM[y0:y1, x0:x1]

            # Reconstruct (distance in mm → meters if needed)
            recoIM = vc.recoFrame(cropIM, distance * 1e-3)

            # === LAYOUT FOR IMAGES SIDE BY SIDE ===
            # Create four columns: [1, 6, 6, 1] to allow wide space for each image
            col_left, col_img1, col_img2, col_right = st.columns([1, 6, 6, 1])
            with col_img1:
                # Set a fixed width so the image is large
                st.image(grayIM, caption="Full Image", channels="GRAY", width=600)
            with col_img2:
                st.image(recoIM, caption="Reconstructed Image", channels="GRAY", width=600)

            # --- SELECT CROP CENTER ---
            # Button to toggle interactive selection
            if st.button("Select Crop Center"):
                st.session_state["select_mode"] = not st.session_state["select_mode"]

            if st.session_state["select_mode"]:
                st.info("Click on the Full Image below to pick a new crop center.")
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",
                    stroke_width=2,
                    background_image=Image.fromarray(grayIM),
                    update_streamlit=True,
                    height=grayIM.shape[0],
                    width=grayIM.shape[1],
                    drawing_mode="point",  # We'll just record single-click points
                    point_display_radius=6,
                    key="canvas",
                )
                # If user clicked, retrieve the last point
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if objects:
                        last_obj = objects[-1]
                        new_x_center = int(last_obj["left"])
                        new_y_center = int(last_obj["top"])
                        st.success(f"New center: (x={new_x_center}, y={new_y_center})")
                        st.session_state["crop_center"] = (new_x_center, new_y_center)

            # --- SAVE IMAGES BUTTON ---
            if st.button("Save Images"):
                # Your logic to save images
                st.success("Images saved successfully!")

    # Clean up temporary file
    os.remove(temp_video_path)
