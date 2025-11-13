import os
import cv2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def read_nc_file(
    path_to_file: str,
    data_variable: str,
) -> np.ndarray:
    """
    Lê um volume a partir de um arquivo .nc.

    Args:
        path_to_file (str): Caminho absoluto para o arquivo .nc;
    """
    ds = xr.open_dataset(path_to_file, engine="h5netcdf")
    img_array = ds[data_variable].T.values
    return img_array

def lazzy_read_nc_file(
    path_to_file: str,
    data_variable: str,
) -> np.ndarray:
    with xr.open_dataset(path_to_file, engine="h5netcdf") as ds:
        # data = ds[data_variable].astype("float32")
        data = ds[data_variable]
        img_array = data.transpose().values

    return img_array

def find_nc_files(base_dir: str):
    """
    Varre a base e retorna um dicionário com os caminhos dos arquivos .nc
    levando em conta as resoluções específicas de Pxx e SWxx.
    """
    all_nc = {}

    # Percorre as subpastas (P01, P02, SW01, etc.)
    subfolders = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and f != "origin" and f  != "central_volumes"
    ]

    # print(subfolders)

    for sub in subfolders:
        name = os.path.basename(sub)
        files = [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".nc")]
        # print(files)

        # Detecta tipo de volume e define resoluções correspondentes
        if name.startswith("P"):
            res_high, res_low = "08000", "64000"
        elif name.startswith("SW"):
            res_high, res_low = "06000", "48000"

        all_nc[name] = {
            "vol_low": next((f for f in files if f"_resolution_{res_low}nm.nc" in f and "segmented" not in f), None),
            "mask_low": next((f for f in files if f"_resolution_{res_low}nm_segmented.nc" in f), None),
            "vol_high": next((f for f in files if f"_resolution_{res_high}nm.nc" in f and "segmented" not in f), None),
            "mask_high": next((f for f in files if f"_resolution_{res_high}nm_segmented.nc" in f), None),
        }

    return all_nc

def extract_central_cube(volume_path: str, output_path: str, data_var: str = "data", size: int = 500):
    """
    Extrai um cubo central de tamanho `size³` de um volume .nc e salva em um novo arquivo.

    Args:
        volume_path (str): Caminho do arquivo de entrada (.nc)
        output_path (str): Caminho de saída para salvar o cubo central (.nc)
        data_var (str): Nome da variável de dados no arquivo
        size (int): Tamanho da aresta do cubo (default = 500)
    """
    print("Processando:", volume_path)

    output_path = os.path.splitext(output_path)[0] + ".npy"
    
    with xr.open_dataset(volume_path, engine="h5netcdf") as ds:
        # data = ds[data_var].astype("float32")
        data = ds[data_var]
        z, y, x = data.shape
        cz, cy, cx = z // 2, y // 2, x // 2
        half = size // 2
        z0, z1 = max(0, cz - half), min(z, cz + half)
        y0, y1 = max(0, cy - half), min(y, cy + half)
        x0, x1 = max(0, cx - half), min(x, cx + half)
        central_cube = data[z0:z1, y0:y1, x0:x1]
        central_cube = np.array(central_cube, dtype=np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, central_cube)
    print(f"✅ Cubo central salvo em: {output_path}")

def plot_middle_slices(volume_paths: dict, name: str, data_var="data", save_dir="plots/volumes_and_masks"):
    """Plota fatias centrais dos volumes e suas máscaras."""

    os.makedirs(save_dir, exist_ok=True)

    vol_low = lazzy_read_nc_file(volume_paths["vol_low"], data_var)
    mask_low = lazzy_read_nc_file(volume_paths["mask_low"], data_var)
    vol_high = lazzy_read_nc_file(volume_paths["vol_high"], data_var)
    mask_high = lazzy_read_nc_file(volume_paths["mask_high"], data_var)

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
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"✅ Figura salva em: {save_path}")


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


def central_crop(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Retorna o recorte central de uma imagem numpy.

    Args:
        image (np.ndarray): Imagem de entrada (2D ou 3D, se colorida).
        size (tuple): Tupla (width, height) do recorte desejado.

    Returns:
        np.ndarray: Recorte central da imagem.
    """
    height, width = image.shape[:2]
    crop_width, crop_height = size

    if crop_width > width or crop_height > height:
        raise ValueError("O tamanho do recorte é maior que a dimensão da imagem.")

    # Calcula os índices de início e fim do recorte
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2

    end_x = start_x + crop_width
    end_y = start_y + crop_height

    # Faz o recorte
    if image.ndim == 2:  # grayscale
        cropped = image[start_y:end_y, start_x:end_x]
    else:  # color ou multi-channel
        cropped = image[start_y:end_y, start_x:end_x, :]

    return cropped
    