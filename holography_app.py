import streamlit as st
import reco as vc  # Your updated reconstruction code
import cv2
import numpy as np
import tempfile
import os

# Wide layout for side-by-side images
st.set_page_config(layout="wide")
st.title("Holographic Reconstruction Demo")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    uploaded_video = st.file_uploader("Upload Hologram Video", type=["mp4", "avi", "mov"])
    frame_number = st.number_input("Frame #", min_value=0, value=0, step=1)
    crop_size = st.slider("Crop (Half-Width in Pixels)", min_value=200, max_value=400, value=10)
    z_distance = st.slider("Z Distance (mm)", min_value=1, max_value=1000, value=267)
    st.write("Adjust to find best focus.")

# Helper to clamp index ranges
def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# --- MAIN LOGIC ---
if uploaded_video is not None:
    # Write the uploaded video to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        temp_video_path = tmp.name

    # Open the video
    cap = vc.openVid(temp_video_path)
    if not cap or not cap.isOpened():
        st.error("Error opening video file.")
    else:
        # Ensure frame_number is in valid range
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = clamp(frame_number, 0, total_frames - 1)

        ret, raw_frame = vc.getFrame(cap, frame_index)
        if not ret:
            st.error(f"Could not read frame {frame_index}.")
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Center coords
            cx, cy = w // 2, h // 2

            # Draw a crosshair for clarity
            colorIM = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(
                colorIM,
                (cx, cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2
            )

            # Crop around center
            x0 = clamp(cx - crop_size, 0, w)
            x1 = clamp(cx + crop_size, 0, w)
            y0 = clamp(cy - crop_size, 0, h)
            y1 = clamp(cy + crop_size, 0, h)
            crop = gray[y0:y1, x0:x1]

            # Convert Z from mm to meters
            z_m = z_distance * 1e-3
            # Reconstruct
            reco = vc.recoFrame(crop, z_m)

            # (Optional) Resize reconstruction back to original dimension
            reco_resized = cv2.resize(reco, (w, h), interpolation=cv2.INTER_LINEAR)

            # Layout: Original on left, Reconstructed on right
            col1, col2 = st.columns(2)
            with col1:
                st.image(colorIM, caption=f"Frame {frame_index} (Raw Hologram)", channels="BGR")
            with col2:
                st.image(reco_resized, caption=f"Reconstructed @ {z_distance} mm", channels="GRAY")

            # Save button (optional)
            if st.button("Save Images"):
                # Example save logic
                cv2.imwrite("hologram_frame.png", colorIM)
                cv2.imwrite("reconstructed.png", reco_resized)
                st.success("Images saved to current directory.")

    # Clean up
    os.remove(temp_video_path)
