import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from irtk.read import find_nc_files, extract_central_cube

path_do_database = R"D:\data_br_petro"

output_root = R"D:\volume_registration\central_volumes_50"

dataset = find_nc_files(path_do_database)

os.makedirs(output_root, exist_ok=True)


for name, paths in dataset.items():
    if not all(paths.values()):
        print(f"⚠️  Arquivos faltando para {name}, pulando.")
        continue

    vol_high = paths["vol_high"]
    mask_high = paths["mask_high"]

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    vol_out = os.path.join(out_dir, f"{name}_vol_high_central.nc")
    mask_out = os.path.join(out_dir, f"{name}_mask_high_central.nc")

    extract_central_cube(vol_high, vol_out, data_var="data", size=50)
    extract_central_cube(mask_high, mask_out, data_var="data", size=50)