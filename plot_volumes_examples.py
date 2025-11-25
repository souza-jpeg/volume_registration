import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


from irtk.read import find_nc_files
from irtk.utils import plot_middle_slices


path_do_database = R"E:\data_br_petro"

dataset = find_nc_files(path_do_database)

for name, paths in dataset.items():
    if not all(paths.values()):
        print(f"⚠️  Arquivos faltando para {name}, pulando.")
        continue

    print(f"\n📁 Processando {name}...")
    plot_middle_slices(paths, name, data_var="data")