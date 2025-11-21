import os
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split

def get_volumes_folders(
    path_to_dataset: str 
):
    subfolders = []
    for volume_folder in os.listdir(path_to_dataset):
        subfolders.append(volume_folder)
    subfolders.remove('origin')
    return subfolders
    
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

