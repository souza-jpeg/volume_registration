import os
import numpy as np
import xarray as xr
from PIL import Image
from sklearn.model_selection import train_test_split

def create_dataset_folders(base_path):
    if not os.path.exists(base_path):
        subfolders = [
            "train_HR/dataset/SRF_8x/data",
            "train_HR/dataset/SRF_8x/target",
            "val_HR/dataset/SRF_8x/data",
            "val_HR/dataset/SRF_8x/target",
            "test_HR/dataset/SRF_8x/data",
            "test_HR/dataset/SRF_8x/target",
        ]

        for subfolder in subfolders:
            final_path = os.path.join(base_path, subfolder)
            os.makedirs(final_path, exist_ok=True)

        print(f"Estrutura criada em: {base_path}")
    else:
        print(f"O diretório '{base_path}' já existe. Nada foi criado.")

def get_volumes_folders(
    path_to_dataset: str 
):
    subfolders = []
    for volume_folder in os.listdir(path_to_dataset):
        subfolders.append(volume_folder)
    subfolders.remove('origin')
    return subfolders

def get_files_in_subfolder(
  path_to_subfolder: str      
) -> list[str]:
    files_path = []
    for file in os.listdir(path_to_subfolder):
        if not file.endswith("_segmented.nc"):
            files_path.append(os.path.join(path_to_subfolder, file))
    return files_path
    
def get_dataset_random_split(
    volumes_list: list[str],
    test_size: float,
    random_seed: int, 
    do_validation_split: bool = False
):
    train_split, test_split = train_test_split(
        volumes_list,
        test_size=test_size,
        random_state=random_seed
    )

    if do_validation_split:
        train_split, val_split = train_test_split(
            train_split,
            test_size=test_size,
            random_state=random_seed      
        )
        return train_split, val_split, test_split
    
    return train_split, test_split

def central_crop(
    img_array: np.ndarray, 
    crop_shape: tuple, 
):
    h, w = img_array.shape[:2]
    ch, cw = crop_shape
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    end_h = start_h + ch
    end_w = start_w + cw
    cropped = img_array[start_h:end_h, start_w:end_w]
    return cropped

