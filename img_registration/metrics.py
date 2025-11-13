import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(a: np.ndarray, b: np.ndarray):
    return np.mean((a - b) ** 2)

def ncc(a: np.ndarray, b: np.ndarray):
    a_norm = (a - np.mean(a)) / (np.std(a) + 1e-8)
    b_norm = (b - np.mean(b)) / (np.std(b) + 1e-8)
    return np.mean(a_norm * b_norm)

def ssim_metric(a: np.ndarray, b: np.ndarray):
    mid = a.shape[2] // 2
    return ssim(a[:, :, mid], b[:, :, mid], data_range=b.max() - b.min())

def evaluate_registration(fixed, registered):
    return {
        "MSE": mse(fixed, registered),
        "NCC": ncc(fixed, registered),
        "SSIM": ssim_metric(fixed, registered),
    }
