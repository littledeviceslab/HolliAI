import cv2
import numpy as np

def openVid(vid):
    cap = cv2.VideoCapture(vid)
    return cap

def getFrame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, rawFrame = cap.read()
    return ret, rawFrame

def propagate(input_img, wvlen, zdist, dxy):
    """
    Fresnel propagation of input_img by distance zdist using wavelength wvlen
    and pixel size dxy. input_img should be a 2D array (amplitude).
    """
    M, N = input_img.shape
    # Frequency grids
    fx = np.fft.fftfreq(N, d=dxy)
    fy = np.fft.fftfreq(M, d=dxy)
    FX, FY = np.meshgrid(fx, fy)
    kxy2 = FX**2 + FY**2

    # Fourier transform (with shift so center is at 0 freq)
    E0 = np.fft.fft2(np.fft.fftshift(input_img))

    # Fresnel kernel
    H = np.exp(-1j * np.pi * wvlen * zdist * kxy2)

    # Inverse FFT to get propagated field
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * H))
    return output_img

def recoFrame(cropIM, z):
    """
    Reconstruct the cropped hologram at distance z (in meters).
    """
    # Convert to float [0..1]
    cropIM_float = cropIM.astype(np.float32) / 255.0
    # Estimate amplitude (assuming cropIM is intensity)
    amp0 = np.sqrt(cropIM_float)

    # System parameters (update to match your actual setup)
    dxy   = 1.4e-6      # Pixel size in meters (e.g. 1.4 μm)
    wvlen = 650.0e-9    # Wavelength in meters (e.g. 650 nm)

    # Propagate
    field = propagate(amp0, wvlen, z, dxy)
    # Intensity = |field|^2
    intensity = np.abs(field)**2

    # Normalize to [0..255]
    if np.max(intensity) > 0:
        intensity_scaled = 255.0 * intensity / np.max(intensity)
    else:
        intensity_scaled = intensity

    return intensity_scaled.astype(np.uint8)
