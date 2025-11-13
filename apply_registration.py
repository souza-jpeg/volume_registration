import itk
import numpy as np
import matplotlib.pyplot as plt
from read import read_nc_file, lazzy_read_nc_file, central_crop
from img_registration.rigid import register_volumes, simple_register_volumes, register_images
from img_registration.metrics import evaluate_registration

fixed_path = R"D:\data_br_petro\P01\P01_dimensions_186_168_150_resolution_64000nm.nc"
# moving_path = R"D:\data_br_petro\P01\P01_dimensions_1488_1340_1200_resolution_08000nm.nc"
moving_path = R"D:\volume_registration\central_volumes_50\P01\P01_vol_high_central.npy"



fixed = lazzy_read_nc_file(fixed_path, "data")
# moving = lazzy_read_nc_file(moving_path, "data")
moving = np.load(moving_path)

fixed_slice = fixed[fixed.shape[0] // 2, :, :]
moving_slice = moving[moving.shape[0] // 2, :, :]
moving_slice_cropped = moving[moving.shape[0] // 2, :, :]


# moving_slice_cropped = central_crop(moving_slice, (64,64))
# moving = np.load(moving_path)

fixed_slice = itk.image_from_array(fixed_slice.astype(np.float32))
moving_slice = itk.image_from_array(moving_slice.astype(np.float32))

# fixed = itk.image_from_array(fixed)
# moving = itk.image_from_array(moving)

print("dimensão do fixed: ", fixed_slice.shape, fixed_slice.dtype, type(fixed_slice))
print("dimensão do moving: ", moving_slice.shape, moving_slice.dtype, type(moving_slice))
print("dimensão do cropped: ", moving_slice_cropped.shape, moving_slice_cropped.dtype, type(moving_slice_cropped))

registered, transform = register_images(fixed_slice, moving_slice)

# fixed_slice = fixed[fixed.shape[0] // 2, :, :]
# moving_slice = moving[moving.shape[0] // 2, :, :]
# registered_slice = registered[registered.shape[0] // 2, :, :]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(moving_slice, cmap="gray")
axes[0].set_title("Moving (original)")
axes[1].imshow(fixed_slice, cmap="gray")
axes[1].set_title("Fixed (referência)")
axes[2].imshow(registered, cmap="gray")
axes[2].set_title("Moving (registrado)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
