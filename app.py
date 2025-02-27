"""
Interactive Holographic Reconstruction with Streamlit Interface
Based on the original work by Thomas Zimmerman, IBM Research-Almaden
Holographic Reconstruction Algorithms by Nick Antipac, UC Berkeley and Daniel Elnatan, UCSF

This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, Center for Cellular Construction.
Disclaimer: Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

This version uses PIL instead of OpenCV for improved cloud compatibility.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import os

# Reconstruction functions from reco.py
def propagate(input_img, wvlen, zdist, dxy):
    M, N = input_img.shape  # get image size, rows M, columns N, they must be even numbers!
    
    # Make sure M and N are even numbers
    if M % 2 != 0:
        input_img = input_img[:-1, :]
        M -= 1
    if N % 2 != 0:
        input_img = input_img[:, :-1]
        N -= 1
    
    # prepare grid in frequency space with origin at 0,0
    _x1 = np.arange(0, N/2)
    _x2 = np.arange(N/2, 0, -1)
    _y1 = np.arange(0, M/2)
    _y2 = np.arange(M/2, 0, -1)
    _x = np.concatenate([_x1, _x2])
    _y = np.concatenate([_y1, _y2])
    x, y = np.meshgrid(_x, _y)
    kx, ky = x / (dxy * N), y / (dxy * M)
    kxy2 = (kx * kx) + (ky * ky)

    # compute FT at z=0
    E0 = np.fft.fft2(np.fft.fftshift(input_img))

    # compute phase aberration 
    _ph_abbr = np.exp(-1j * np.pi * wvlen * zdist * kxy2)
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * _ph_abbr))
    return output_img

def recoFrame(cropIM, z): 
    dxy = 1.4e-6  # imager pixel (meters)
    wvlen = 650.0e-9  # wavelength of light is red, 650 nm
    
    # Ensure the crop image has dimensions that are even numbers
    h, w = cropIM.shape
    if h % 2 != 0:
        cropIM = cropIM[:-1, :]
    if w % 2 != 0:
        cropIM = cropIM[:, :-1]
        
    res = propagate(np.sqrt(cropIM), wvlen, z, dxy)  # calculate wavefront at z
    amp = np.abs(res)**2  # output is the complex field, still need to compute intensity via abs(res)**2
    # Normalize to 0-255 range
    amp_norm = (amp - amp.min()) / (amp.max() - amp.min()) * 255
    ampInt = amp_norm.astype('uint8')  
    return ampInt

def clamp(value, minv, maxv):
    return max(min(value, maxv), minv)

# PIL-based video frame extraction
def extract_frame(file_path, frame_index):
    import subprocess
    import sys
    
    try:
        # Try to install ffmpeg-python if not already installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        import ffmpeg
    except Exception as e:
        st.error(f"Could not install or import ffmpeg-python: {e}")
        return None
    
    try:
        # Create a temporary file for the extracted frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            frame_filename = tmp_file.name
        
        # Use ffmpeg to extract the specific frame
        (
            ffmpeg
            .input(file_path, ss=frame_index/30)  # Assuming 30fps, adjust if different
            .output(frame_filename, vframes=1)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        # Read the extracted frame
        with Image.open(frame_filename) as img:
            # Convert to grayscale and return as numpy array
            gray_img = img.convert('L')
            frame_array = np.array(gray_img)
        
        # Clean up temporary file
        os.unlink(frame_filename)
        
        return frame_array
    except Exception as e:
        st.error(f"Error extracting frame: {e}")
        return None

# Alternative frame extraction using PIL directly (for some video formats)
def extract_frame_pil(file_path, frame_index):
    try:
        # Try to install Pillow-SIMD for better performance
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        
        # Open the video file
        with Image.open(file_path) as img:
            # Some formats support seeking to a specific frame
            if hasattr(img, 'seek') and hasattr(img, 'n_frames'):
                frame_to_use = min(frame_index, img.n_frames - 1)
                img.seek(frame_to_use)
                gray_img = img.convert('L')
                return np.array(gray_img)
        
        # If we can't use PIL directly, return None
        return None
    except Exception as e:
        # Return None if there's an error, we'll fall back to ffmpeg
        return None

# Initialize session state for variables
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'z_value' not in st.session_state:
    st.session_state.z_value = 64
if 'crop_size' not in st.session_state:
    st.session_state.crop_size = 25
if 'display_scale' not in st.session_state:
    st.session_state.display_scale = 1
if 'center_x' not in st.session_state:
    st.session_state.center_x = 960  # Default center (half of 1920)
if 'center_y' not in st.session_state:
    st.session_state.center_y = 540  # Default center (half of 1080)
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'max_frames' not in st.session_state:
    st.session_state.max_frames = 100  # Default, will try to detect actual count
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'reconstructed_image' not in st.session_state:
    st.session_state.reconstructed_image = None
if 'crop_image' not in st.session_state:
    st.session_state.crop_image = None
if 'window' not in st.session_state:
    st.session_state.window = [0, 1080, 0, 1920]

# Constants from original code
Z_SCALE = 0.00001  # convert integer Z units to 10 um
WINDOW_SCALE = 10  # window size increment
FULL_SCALE = 2     # reduce full scale image by this factor so it fits in window

def update_window():
    xc = st.session_state.center_x
    yc = st.session_state.center_y
    CROP = st.session_state.crop_size
    
    x0 = xc - (WINDOW_SCALE * CROP)
    x1 = xc + (WINDOW_SCALE * CROP)
    y0 = yc - (WINDOW_SCALE * CROP)
    y1 = yc + (WINDOW_SCALE * CROP)

    # Get video resolution from the current frame if available
    if st.session_state.current_frame is not None:
        yRez, xRez = st.session_state.current_frame.shape
    else:
        xRez, yRez = 1920, 1080  # Default

    x0 = clamp(x0, 0, xRez)
    x1 = clamp(x1, x0, xRez)
    y0 = clamp(y0, 0, yRez)
    y1 = clamp(y1, y0, yRez)

    st.session_state.window = [int(y0), int(y1), int(x0), int(x1)]

def process_image():
    if st.session_state.video_path is None:
        st.warning("Please upload a video first.")
        return
    
    # Update the crop window
    update_window()
    
    # Get the current frame
    frame_data = extract_frame(st.session_state.video_path, st.session_state.frame_count)
    
    if frame_data is None:
        # Try alternative method
        frame_data = extract_frame_pil(st.session_state.video_path, st.session_state.frame_count)
        
    if frame_data is None:
        st.error("Error reading frame. The uploaded file may not be a supported video format.")
        return
    
    # Store current frame
    st.session_state.current_frame = frame_data
    
    # Crop the image
    window = st.session_state.window
    # Make sure window indices are within bounds
    if (window[0] >= frame_data.shape[0] or 
        window[1] > frame_data.shape[0] or 
        window[2] >= frame_data.shape[1] or 
        window[3] > frame_data.shape[1]):
        # Adjust window to safe values
        window = [0, min(1080, frame_data.shape[0]), 0, min(1920, frame_data.shape[1])]
        st.session_state.window = window
        
    crop_image = frame_data[window[0]:window[1], window[2]:window[3]]
    
    # Store the cropped image
    st.session_state.crop_image = crop_image
    
    # Reconstruct the image
    z_value = st.session_state.z_value * Z_SCALE
    reconstructed_image = recoFrame(crop_image, z_value)
    
    # Store the reconstructed image
    st.session_state.reconstructed_image = reconstructed_image

def get_image_download_link(img, filename, text):
    """Generate a link to download an image"""
    buffered = io.BytesIO()
    img_pil = Image.fromarray(img)
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def count_frames(video_path):
    """Estimate number of frames in video using ffmpeg"""
    try:
        import subprocess
        import json
        
        # Install ffmpeg-python if needed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg-python"], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
        import ffmpeg
        
        # Get video information
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream and 'nb_frames' in video_stream:
            return int(video_stream['nb_frames'])
        elif video_stream and 'duration' in video_stream and 'r_frame_rate' in video_stream:
            # Calculate from duration and frame rate
            duration = float(video_stream['duration'])
            fps_parts = video_stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) > 1 else float(fps_parts[0])
            return int(duration * fps)
        return 100  # Default
    except Exception as e:
        st.warning(f"Could not determine frame count: {e}")
        return 100  # Default to 100 frames

# Main Streamlit app
st.title("Interactive Holographic Reconstruction")

st.markdown("""
### About
This application reconstructs holographic images from video frames.
Upload a video file, navigate through frames, adjust the crop area and Z-value,
and save the reconstructed images.

### Instructions
1. Upload a video file (.mp4)
2. Use the frame navigation buttons to find frames of interest
3. Adjust the center position to target your region of interest
4. Adjust the crop size to capture the object and its fringes
5. Adjust the Z value to focus the reconstruction
6. Save the reconstructed image when satisfied
""")

# Upload video file
uploaded_file = st.file_uploader("Upload Video (MP4)", type=["mp4", "gif", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    # Update the video path in session state
    st.session_state.video_path = tfile.name
    
    # Try to get video information and frame count
    st.session_state.max_frames = count_frames(tfile.name)
    
    # Try to get first frame to determine dimensions
    first_frame = extract_frame(tfile.name, 0)
    if first_frame is not None:
        height, width = first_frame.shape
        
        # Update default center position based on video dimensions
        st.session_state.center_x = width // 2
        st.session_state.center_y = height // 2
        
        st.write(f"Video loaded: {width}x{height}, approximately {st.session_state.max_frames} frames")
    else:
        st.write(f"Video loaded. Could not determine dimensions. Estimated frames: {st.session_state.max_frames}")
    
    # Process the first frame automatically
    process_image()

# Frame navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Frame -10"):
        st.session_state.frame_count = max(0, st.session_state.frame_count - 10)
        process_image()
with col2:
    if st.button("Frame -1"):
        st.session_state.frame_count = max(0, st.session_state.frame_count - 1)
        process_image()
with col3:
    if st.button("Frame +1"):
        st.session_state.frame_count = min(st.session_state.max_frames - 1, st.session_state.frame_count + 1)
        process_image()
with col4:
    if st.button("Frame +10"):
        st.session_state.frame_count = min(st.session_state.max_frames - 1, st.session_state.frame_count + 10)
        process_image()

# Create a frame slider
frame_slider = st.slider("Frame", 0, max(0, st.session_state.max_frames - 1), st.session_state.frame_count)
if frame_slider != st.session_state.frame_count:
    st.session_state.frame_count = frame_slider
    process_image()

# Display status
st.write(f"Frame: {st.session_state.frame_count} / {st.session_state.max_frames-1} | "
         f"Crop Size: {st.session_state.crop_size} | "
         f"Z: {st.session_state.z_value} | "
         f"Center: ({st.session_state.center_x}, {st.session_state.center_y})")

# Center position adjustment
col1, col2 = st.columns(2)
with col1:
    center_x = st.number_input("Center X", value=st.session_state.center_x, step=2)
    if center_x != st.session_state.center_x:
        st.session_state.center_x = center_x if center_x % 2 == 0 else center_x + 1
        process_image()
        
with col2:
    center_y = st.number_input("Center Y", value=st.session_state.center_y, step=2)
    if center_y != st.session_state.center_y:
        st.session_state.center_y = center_y if center_y % 2 == 0 else center_y + 1
        process_image()

# Crop size adjustment
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Crop -10"):
        st.session_state.crop_size = max(1, st.session_state.crop_size - 10)
        process_image()
with col2:
    if st.button("Crop -1"):
        st.session_state.crop_size = max(1, st.session_state.crop_size - 1)
        process_image()
with col3:
    if st.button("Crop +1"):
        st.session_state.crop_size += 1
        process_image()
with col4:
    if st.button("Crop +10"):
        st.session_state.crop_size += 10
        process_image()

# Crop size slider
crop_slider = st.slider("Crop Size", 1, 100, st.session_state.crop_size)
if crop_slider != st.session_state.crop_size:
    st.session_state.crop_size = crop_slider
    process_image()

# Z value adjustment
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Z -10"):
        st.session_state.z_value = max(1, st.session_state.z_value - 10)
        process_image()
with col2:
    if st.button("Z -1"):
        st.session_state.z_value = max(1, st.session_state.z_value - 1)
        process_image()
with col3:
    if st.button("Z +1"):
        st.session_state.z_value += 1
        process_image()
with col4:
    if st.button("Z +10"):
        st.session_state.z_value += 10
        process_image()

# Z value slider
z_slider = st.slider("Z Value", 1, 200, st.session_state.z_value)
if z_slider != st.session_state.z_value:
    st.session_state.z_value = z_slider
    process_image()

# Display the full frame
if st.session_state.current_frame is not None:
    st.subheader("Full Image")
    # Resize for display
    h, w = st.session_state.current_frame.shape
    resized_h, resized_w = h // FULL_SCALE, w // FULL_SCALE
    full_img_display = np.array(Image.fromarray(st.session_state.current_frame).resize((resized_w, resized_h)))
    
    # Display the image
    st.image(full_img_display, use_column_width=True)
    
    # Display crop window information
    st.write("Crop Window:")
    st.write(f"Top: {st.session_state.window[0]}, Bottom: {st.session_state.window[1]}, "
             f"Left: {st.session_state.window[2]}, Right: {st.session_state.window[3]}")

# Display reconstruction
if st.session_state.reconstructed_image is not None and st.session_state.crop_image is not None:
    st.subheader("Reconstructed Image")
    
    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.write("Cropped Raw Image")
        st.image(st.session_state.crop_image, use_column_width=True)
    with col2:
        st.write("Reconstructed Image")
        st.image(st.session_state.reconstructed_image, use_column_width=True)
    
    # Save buttons
    col1, col2 = st.columns(2)
    with col1:
        # Create a download link for the raw cropped image
        if st.button("Save Raw Image"):
            raw_filename = f"frame_{st.session_state.frame_count}_x{st.session_state.center_x}_y{st.session_state.center_y}_z{st.session_state.z_value*10}_raw.jpg"
            st.markdown(get_image_download_link(st.session_state.crop_image, raw_filename, f"Download {raw_filename}"), unsafe_allow_html=True)
    
    with col2:
        # Create a download link for the reconstructed image
        if st.button("Save Reconstructed Image"):
            reco_filename = f"frame_{st.session_state.frame_count}_x{st.session_state.center_x}_y{st.session_state.center_y}_z{st.session_state.z_value*10}_holo.jpg"
            st.markdown(get_image_download_link(st.session_state.reconstructed_image, reco_filename, f"Download {reco_filename}"), unsafe_allow_html=True)

# Clean up temporary file when the app is done
if st.session_state.video_path is not None and 'tfile' in locals():
    try:
        os.unlink(tfile.name)
    except:
        pass

st.markdown("""
### Credits
- Original code by Thomas Zimmerman, IBM Research-Almaden
- Holographic Reconstruction Algorithms by Nick Antipac, UC Berkeley and Daniel Elnatan, UCSF
- This work is funded by the National Science Foundation (NSF) grant No. DBI-1548297, Center for Cellular Construction.
- Disclaimer: Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.
""")
