from irtk.upscalling.esrgan.generate_png_data import get_volumes_folders, get_dataset_random_split

volume_folders = get_volumes_folders(R'D:\data_br_petro')
print(volume_folders)

train, val, test = get_dataset_random_split(
    volume_folders, 
    0.2,
    21,
    True
)

print(train)
print(val)
print(test)