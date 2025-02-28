import cv2
import numpy as np

def openVid(vid):
    cap = cv2.VideoCapture(vid)
    return cap

def getFrame(cap, index):
    cap.set(1, index)
    ret, rawFrame = cap.read()
    return ret, rawFrame

def propagate(input_img, wvlen, zdist, dxy):
    M, N = input_img.shape  # Assumes M and N are even
    # Create spatial frequency grids robustly using fftfreq
    fx = np.fft.fftfreq(N, d=dxy)
    fy = np.fft.fftfreq(M, d=dxy)
    FX, FY = np.meshgrid(fx, fy)
    kxy2 = FX**2 + FY**2

    # Compute the Fourier transform with centering
    E0 = np.fft.fft2(np.fft.fftshift(input_img))
    # Fresnel propagation kernel
    H = np.exp(-1j * np.pi * wvlen * zdist * kxy2)
    # Propagate the field
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * H))
    return output_img

def recoFrame(cropIM, z):
    # Convert the cropped image to float and normalize to [0,1]
    cropIM_norm = cropIM.astype(np.float32) / 255.0
    # Estimate the amplitude by taking the square root
    amp0 = np.sqrt(cropIM_norm)
    
    # Fixed system parameters (ensure these match your setup)
    dxy   = 1.4e-6       # Pixel size in meters
    wvlen = 650.0e-9     # Wavelength in meters
    
    # z should be provided in meters (conversion done by caller)
    res = propagate(amp0, wvlen, z, dxy)
    
    # Compute intensity as the squared magnitude of the propagated field
    intensity = np.abs(res)**2
    # Normalize intensity to the full 0-255 range
    if np.max(intensity) > 0:
        intensity_norm = 255 * intensity / np.max(intensity)
    else:
        intensity_norm = intensity
    return intensity_norm.astype('uint8')
