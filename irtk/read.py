import os
import numpy as np
import xarray as xr

def read_nc_file(
    path_to_file: str,
    data_variable: str,
) -> np.ndarray:
    with xr.open_dataset(path_to_file, engine="h5netcdf") as ds:
        data = ds[data_variable]
        img_array = data.transpose().values
    return img_array

def find_nc_files(path_to_dataset: str):
    all_nc = {}
    # Percorre as subpastas (P01, P02, SW01, etc.)
    subfolders = [
        os.path.join(path_to_dataset, f)
        for f in os.listdir(path_to_dataset)
        if os.path.isdir(os.path.join(path_to_dataset, f)) and f != "origin" and f  != "central_volumes"
    ]

    for sub in subfolders:
        name = os.path.basename(sub)
        files = [os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".nc")]

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


def extract_central_cube(
        path_to_volume: str, 
        output_path: str, 
        data_var: str = "data", 
        size: int = 500
    ):
    """
    Args:
        path_to_volume (str): Caminho do arquivo de entrada (.nc)
        output_path (str): Caminho de saída para salvar o cubo central (.nc)
        data_var (str): Nome da variável de dados no arquivo
        size (int): Tamanho da aresta do cubo (default = 500)
    """
    print("Processando:", path_to_volume)

    output_path = os.path.splitext(output_path)[0] + ".npy"
    
    with xr.open_dataset(path_to_volume, engine="h5netcdf") as ds:
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
    print(f"Cubo central salvo em: {output_path}")