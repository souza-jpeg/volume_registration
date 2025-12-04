import os
import cv2
import torch
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from irtk.read import read_nc_file



def save_images_float32_normalized(
    data: torch.Tensor, 
    target: torch.Tensor,
    epoch: int, 
    i: int,
    path_teste = str
):
    batch_size = data.shape[0]

    for j in range(batch_size):
        lr_img = data[j]   
        hr_img = target[j] 

        lr_img_norm = (lr_img.float() / 65535.).clamp(0,1)
        hr_img_norm = (hr_img.float() / 65535.).clamp(0,1)

        lr_name = f"epoch{epoch}_batch{i}_idx{j}_LR.png"
        hr_name = f"epoch{epoch}_batch{i}_idx{j}_HR.png"

        save_image(lr_img_norm, os.path.join(os.path.join(path_teste, 'lr', lr_name)))
        save_image(hr_img_norm, os.path.join(os.path.join(path_teste, 'hr', hr_name)))


def save_all_volumes_slices_in_path(
    volume: np.ndarray, 
    path_to_output: str
):
    os.makedirs(path_to_output, exist_ok=True)
    volume = np.asarray(volume)

    for i in range(volume.shape[0]):
        slice_i = volume[i]

        plt.imsave(
            os.path.join(path_to_output, f"slice_{i:04d}.png"),
            slice_i,
            cmap="gray"
        )

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
    return cv2.convertScaleAbs(volume, alpha=(255.0/float(volume.max())))

def get_depth_channel_first(
    volume: np.ndarray
) -> np.ndarray:
    return np.transpose(volume, (2, 0, 1))

def convert_volume_to_rgb(
    volume: np.ndarray
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Wait a 3D array with shape (D,H,W), but got {volume.shape}")
    return np.repeat(volume[..., None], 3, axis=-1)

def get_volume_information(
    volume: np.ndarray
):
    print("Shape do volume: ", volume.shape)
    print("Dtype do volume: ", volume.dtype)
    print(f"Intensidades do volume: Entre {volume.min()} e {volume.max()}")

def get_pytorch_gpu_status():
    import torch
    print("Pytorch GPU recognized? ", torch.cuda.is_available())
    print("Pytorch Device Count: ", torch.cuda.device_count())
    print("First Pytorch device name:", torch.cuda.get_device_name(0))