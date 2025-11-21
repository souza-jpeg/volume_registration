import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from irtk.metrics import mae, mse, psnr, ssim, absolute_difference
from irtk.read import find_nc_files, read_nc_file
from irtk.utils import save_all_volumes_slices_in_path, get_volume_information
from irtk.upscalling.classic import resample_volume_to_resolution

path_do_database = R"D:\data_br_petro"

path_output = R"D:\volume_registration\plot_P01_resolutions"
path_output_real = os.path.join(path_output, 'real_resolutions')
path_output_interpolated = os.path.join(path_output, 'interpolated')

dataset = find_nc_files(path_do_database)
data_volume = dataset["P01"]
path_vol_real_high = data_volume["vol_high"]
path_vol_real_low = data_volume["vol_low"]

# plotando resoluções reais
vol_high = read_nc_file(path_vol_real_high, "data")
vol_low = read_nc_file(path_vol_real_low, "data")

print("volume high")
get_volume_information(vol_high)

print("volume low")
get_volume_information(vol_low)

print(vol_high.shape)
print(vol_low.shape)

save_all_volumes_slices_in_path(vol_high, os.path.join(path_output_real, "high"))
save_all_volumes_slices_in_path(vol_low, os.path.join(path_output_real, "low"))


# gerando interpolação de Lanczos 
interpolated_high_volume = resample_volume_to_resolution(
    volume=vol_high,
    name_volume='PO1',
    current_resolution_nm=(8000, 8000, 8000),
    target_resolution_nm=(8000, 64000, 64000)
)

interpolated_low_volume = resample_volume_to_resolution(
    volume=vol_low,
    name_volume='PO1',
    current_resolution_nm=(64000, 64000, 64000),
    target_resolution_nm=(8000, 8000, 8000)
)


print("volume high interpolated")
get_volume_information(interpolated_high_volume)

print("volume low interpolated")
get_volume_information(interpolated_low_volume)


# o volume grande vai ficar pequeno
save_all_volumes_slices_in_path(interpolated_high_volume, os.path.join(path_output_interpolated, "high"))

#o volume pequeno vai ficar grande
save_all_volumes_slices_in_path(interpolated_low_volume, os.path.join(path_output_interpolated, "low"))



# calculo de metricas
vol_high_original = vol_high[0]
vol_low_original = vol_low[0]

vol_high_interpolated = interpolated_high_volume[0]
vol_low_interpolated = interpolated_low_volume[0]

data_range = 65535.0  # para uint16
mse_val = mse(vol_high_original, vol_low_interpolated)
mae_val = mae(vol_high_original, vol_low_interpolated)
psnr_val = psnr(vol_high_original, vol_low_interpolated, data_range=data_range)
ssim_val = ssim(vol_high_original, vol_low_interpolated, data_range=data_range)
abs_diff = absolute_difference(vol_high_original, vol_high_original)

# Exibe métricas numéricas
print("Métricas entre ORIGINAL e INTERPOLADA (uint16):")
print(f"  MSE : {mse_val:.4f}")
print(f"  MAE : {mae_val:.4f}")
print(f"  PSNR: {psnr_val:.4f} dB")
print(f"  SSIM: {ssim_val:.6f}")

# Visualizações (uma figura por gráfico, conforme instrução)
# plt.figure()
# plt.imshow(vol_high_original, cmap='gray')
# plt.title("Imagem Original (uint16)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(vol_low_interpolated, cmap='gray')
# plt.title("Imagem Interpolada (uint16)")
# plt.axis('off')
# plt.show()

# plt.figure()
# plt.imshow(abs_diff, cmap='gray')
# plt.title("Mapa de Diferença Absoluta |orig - interp|")
# plt.axis('off')
# plt.show()
