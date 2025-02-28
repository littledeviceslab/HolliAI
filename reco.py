import numpy as np
import cv2

def openVid(vid):
    """Open a video file using OpenCV."""
    cap = cv2.VideoCapture(vid)
    return cap

def getFrame(cap, index):
    """Retrieve a specific frame from the video."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, rawFrame = cap.read()
    return ret, rawFrame

def reconstructImage(im, zdist_um):
    """
    Reconstruct the hologram using your manual frequency-grid approach.

    Parameters:
        im         : 2D numpy array (grayscale hologram, 8-bit).
        zdist_um   : Distance (object to sensor) in *microns*.

    Returns:
        imAmp      : Reconstructed image as a uint8 intensity.
    """
    # 1) Convert to float and take sqrt (assuming im is intensity)
    #    NOTE: If 'im' is 8-bit [0..255], we can directly do sqrt(im).
    #    If you need normalization, do: im_float = im.astype(np.float32)/255.0, etc.
    input_img = np.sqrt(im.astype(np.float32))

    # 2) Constants
    dxy   = 1.4e-6      # Pixel spacing (meters) = 1.4 microns
    wvlen = 650.0e-9    # Wavelength of light (meters) = 650 nm

    # Convert zdist from microns to meters so units match
    zdist_m = zdist_um * 1e-6

    # 3) Get image size, must be even values ideally
    M, N = input_img.shape

    # 4) Prepare grid in frequency space with origin at (0,0)
    #    The snippet manually splits the range into [0..N/2] and [N/2..0], etc.
    #    We'll replicate it exactly for consistency with your math:
    _x1 = np.arange(0, N/2)
    _x2 = np.arange(N/2, 0, -1)
    _y1 = np.arange(0, M/2)
    _y2 = np.arange(M/2, 0, -1)

    _x = np.concatenate([_x1, _x2])
    _y = np.concatenate([_y1, _y2])

    x, y = np.meshgrid(_x, _y)
    kx, ky = x / (dxy * N), y / (dxy * M)
    kxy2 = (kx**2 + ky**2)

    # 5) Compute FT at z=0
    #    We use ifftshift(input_img) before fft2, per your snippet
    E0 = np.fft.fft2(np.fft.ifftshift(input_img))

    # 6) Compute phase aberration
    _ph_abbr = np.exp(-1j * np.pi * wvlen * zdist_m * kxy2)

    # 7) Propagate + shift back
    output_img = np.fft.fftshift(np.fft.ifft2(E0 * _ph_abbr))

    # 8) Intensity = |output_img|^2
    amp = np.abs(output_img)**2

    # 9) Convert to 8-bit
    #    NOTE: This direct cast may lose dynamic range if 'amp' is large or small.
    imAmp = amp.astype(np.uint8)

    return imAmp
