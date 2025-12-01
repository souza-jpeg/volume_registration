import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from irtk.read import read_nc_file


def central_crop_and_save(img_array: np.ndarray, crop_shape: tuple, save_path: str):
    h, w = img_array.shape[:2]
    ch, cw = crop_shape
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    end_h = start_h + ch
    end_w = start_w + cw
    cropped = img_array[start_h:end_h, start_w:end_w]
    Image.fromarray(cropped).save(save_path)
    return cropped

path_to_volume = R"D:\data_br_petro\P01\P01_dimensions_1488_1340_1200_resolution_08000nm.nc"
volume = read_nc_file(path_to_file=path_to_volume, data_variable="data")
print("shape do volume: ", volume.shape, volume.dtype)
save_path = "recortada.png"
crop_shape = (1024, 1024)

img = volume[0]

cropped_img = central_crop_and_save(img, crop_shape, save_path)
print("shape da original: ", img.shape, img.dtype)
print("shape da recortada: ", cropped_img.shape, cropped_img.dtype)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Recortada")
plt.imshow(cropped_img, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()