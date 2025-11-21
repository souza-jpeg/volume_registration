import numpy as np
import math
import matplotlib.pyplot as plt

def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    return float(np.mean((x - y) ** 2))

def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    return float(np.mean(np.abs(x - y)))

def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
    err = mse(img1, img2)
    if err == 0:
        return float('inf')
    if data_range is None:
        if np.issubdtype(img1.dtype, np.integer):
            info = np.iinfo(img1.dtype)
            data_range = float(info.max - info.min)
        else:
            data_range = 1.0
    return 10.0 * math.log10((data_range ** 2) / err)

def _ssim_fallback_global(img1: np.ndarray, img2: np.ndarray, data_range: float) -> float:
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    K1, K2 = 0.01, 0.03
    L = data_range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = x.var()
    sigma_y2 = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(num / den)

def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
    if data_range is None:
        if np.issubdtype(img1.dtype, np.integer):
            info = np.iinfo(img1.dtype)
            data_range = float(info.max - info.min)
        else:
            data_range = 1.0
    try:
        from skimage.metrics import structural_similarity as ssim_sk
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        return float(ssim_sk(x, y, data_range=data_range))
    except Exception:
        return _ssim_fallback_global(img1, img2, data_range)

def absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    return np.abs(x - y)

# Usage
# # Calcula métricas
# data_range = 65535.0  # para uint16
# mse_val = mse(orig, interp)
# mae_val = mae(orig, interp)
# psnr_val = psnr(orig, interp, data_range=data_range)
# ssim_val = ssim(orig, interp, data_range=data_range)
# abs_diff = absolute_difference(orig, interp)

# # Exibe métricas numéricas
# print("Métricas entre ORIGINAL e INTERPOLADA (uint16):")
# print(f"  MSE : {mse_val:.4f}")
# print(f"  MAE : {mae_val:.4f}")
# print(f"  PSNR: {psnr_val:.4f} dB")
# print(f"  SSIM: {ssim_val:.6f}")

# # Visualizações (uma figura por gráfico, conforme instrução)
# plt.figure()
# plt.imshow(orig, cmap='gray')
# plt.title("Imagem Original (uint16)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(interp, cmap='gray')
# plt.title("Imagem Interpolada (uint16)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(abs_diff, cmap='gray')
# plt.title("Mapa de Diferença Absoluta |orig - interp|")
# plt.axis('off')
# plt.show()
