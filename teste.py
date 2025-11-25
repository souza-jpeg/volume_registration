import os
from PIL import Image
from irtk.read import read_nc_file
from irtk.upscalling.esrgan.generate_png_data import get_volumes_folders, get_dataset_random_split, \
    get_files_in_subfolder, create_dataset_folders

dataset_root_path = R'D:\data_br_petro'
path_to_png_dataset_output = R'D:\volume_registration\ersgan_dataset'

create_dataset_folders(path_to_png_dataset_output)

volume_folders = get_volumes_folders(dataset_root_path)
print(volume_folders)

train_subfolders, val_subfolders, test_subfolders = get_dataset_random_split(
    volume_folders, 
    0.2,
    21,
    True
)

## Creating Training dataset 
for train_subfolder in train_subfolders:
    print(f"===== Salvando imagens do treino =====")
    print(f"volume {train_subfolder}...")
    path_to_subfolder = os.path.join(dataset_root_path, train_subfolder)
    path_to_files = get_files_in_subfolder(path_to_subfolder)
    
    for file_path in path_to_files:
        volume = read_nc_file(
            path_to_file=file_path,
            data_variable="data"
        )

        if '08000nm' in file_path or '06000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\train_HR\dataset\SRF_8x\target"
        elif '64000nm' in file_path or '48000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\train_HR\dataset\SRF_8x\data"

        for i in range(volume.shape[0]):
            slice_i = volume[i]
            slice_i = slice_i.astype("uint16")
            img = Image.fromarray(slice_i, mode="I;16")
            img.save(os.path.join(path_to_save_image, f"{train_subfolder}_{i}.png"))



## Creating Validation dataset 
for val_subfolder in val_subfolders:
    print(f"===== Salvando imagens da validação =====")
    print(f"volume {val_subfolder}...")
    path_to_subfolder = os.path.join(dataset_root_path, val_subfolder)
    path_to_files = get_files_in_subfolder(path_to_subfolder)
    
    for file_path in path_to_files:
        volume = read_nc_file(
            path_to_file=file_path,
            data_variable="data"
        )

        if '08000nm' in file_path or '06000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\val_HR\dataset\SRF_8x\target"
        elif '64000nm' in file_path or '48000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\val_HR\dataset\SRF_8x\data"

        for i in range(volume.shape[0]):
            slice_i = volume[i]
            slice_i = slice_i.astype("uint16")
            img = Image.fromarray(slice_i, mode="I;16")
            img.save(os.path.join(path_to_save_image, f"{train_subfolder}_{i}.png"))


## Creating Test dataset 
for test_subfolder in test_subfolders:
    print(f"===== Salvando imagens do teste =====")
    print(f"volume {test_subfolder}...")
    path_to_subfolder = os.path.join(dataset_root_path, test_subfolder)
    path_to_files = get_files_in_subfolder(path_to_subfolder)
    
    for file_path in path_to_files:
        volume = read_nc_file(
            path_to_file=file_path,
            data_variable="data"
        )

        if '08000nm' in file_path or '06000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\test_HR\dataset\SRF_8x\target"
        elif '64000nm' in file_path or '48000nm' in file_path:
            path_to_save_image = R"D:\volume_registration\ersgan_dataset\test_HR\dataset\SRF_8x\data"

        for i in range(volume.shape[0]):
            slice_i = volume[i]
            slice_i = slice_i.astype("uint16")
            img = Image.fromarray(slice_i, mode="I;16")
            img.save(os.path.join(path_to_save_image, f"{train_subfolder}_{i}.png"))


