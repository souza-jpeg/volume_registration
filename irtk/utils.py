import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from read import read_nc_file

def plot_middle_slices(
    volume_paths: dict, 
    name: str, data_var="data", 
    save_dir="plots/volumes_and_masks"
):

    os.makedirs(save_dir, exist_ok=True)

    vol_low = read_nc_file(volume_paths["vol_low"], data_var)
    mask_low = read_nc_file(volume_paths["mask_low"], data_var)
    vol_high = read_nc_file(volume_paths["vol_high"], data_var)
    mask_high = read_nc_file(volume_paths["mask_high"], data_var)

    mid_low = vol_low.shape[2] // 2
    mid_high = vol_high.shape[2] // 2

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(vol_low[:, :, mid_low], cmap="gray")
    axs[0, 0].set_title(f"{name} - Volume baixa resolução")
    axs[0, 1].imshow(mask_low[:, :, mid_low], cmap="gray")
    axs[0, 1].set_title(f"{name} - Máscara baixa resolução")
    axs[1, 0].imshow(vol_high[:, :, mid_high], cmap="gray")
    axs[1, 0].set_title(f"{name} - Volume alta resolução")
    axs[1, 1].imshow(mask_high[:, :, mid_high], cmap="gray")
    axs[1, 1].set_title(f"{name} - Máscara alta resolução")

    for ax in axs.flat:
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{name}_middle_slices.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Figura salva em: {save_path}")


def convert_uint16_to_uint8(
    volume: np.ndarray
) -> np.ndarray:
    """
    Converte do dtype do  de uint16 para uint8.

    Args:
        volume (np.ndarray): Volume numpy com dtype uint16.

    Returns:
        (np.ndarray): Volume numpy com dtype uint8.
    """
    return cv2.convertScaleAbs(volume, alpha=(255.0/float(volume.max())))

def get_depth_channel_first(
    volume: np.ndarray
) -> np.ndarray:
    """
    Muda a dimensão de profundidade (eixo Z) de um volume
    numpy para a primeira posição do seu shape.

    Args:
        volume (np.ndarray): Volume com shape (Width, Height, Depth).

    Returns:
        (np.ndarray): Volume com shape (Depth, Width, Height).
    """
    return np.transpose(volume, (2, 0, 1))


def convert_volume_to_rgb(
    volume: np.ndarray
) -> np.ndarray:
    """
    Converte um volume em grayscale para RGB;

    Args:
        volume (np.ndarray): Volume numpy em grayscale;

    Returns:
        (np.ndarray): Volume numpy em RGB;
    """
    if volume.ndim != 3:
        raise ValueError(f"Wait a 3D array with shape (D,H,W), but got {volume.shape}")
    return np.repeat(volume[..., None], 3, axis=-1)