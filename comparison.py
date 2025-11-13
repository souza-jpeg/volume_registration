import itk
import numpy as np
import matplotlib.pyplot as plt
from read import read_nc_file, lazzy_read_nc_file
from img_registration.rigid import register_volumes, simple_register_volumes
from img_registration.metrics import evaluate_registration

fixed_path = R"D:\data_br_petro\P01\P01_dimensions_186_168_150_resolution_64000nm.nc"
moving_path = R"D:\data_br_petro\P01\P01_dimensions_1488_1340_1200_resolution_08000nm.nc"
# moving_path = R"D:\volume_registration\central_volumes_50\P01\P01_vol_high_central.npy"

fixed = lazzy_read_nc_file(fixed_path, "data")
# moving = np.load(moving_path)
moving = lazzy_read_nc_file(moving_path, "data")



fixed_slice = fixed[fixed.shape[0] // 2, :, :]
moving_slice = moving[moving.shape[0] // 2, :, :]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(moving_slice, cmap="gray")
axes[0].set_title("Moving (original)")
axes[1].imshow(fixed_slice, cmap="gray")
axes[1].set_title("Fixed (referência)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()