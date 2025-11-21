import os
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize


path_dataset_high_resolution = R'BrPetro_Low_Resolution'
path_new_resolution_destino = R'BrPetro_Low_to_High'
path_new_resolution_resized = R'BrPetro_High_to_Low_Resolution_Resized'

#Em nanômetros
current_resolution = {
'P':  [64000, 64000, 64000],
'S':  [48000, 48000, 48000]
}

target_resolution = {
'P':  [8000, 8000, 8000],
'S':  [48000, 48000, 6000]
}

target_resolution = {}

def generate_new_name(
    act_name: str, 
    volume
) -> str:
    x = len(volume)
    y = len(volume[0])
    z = len(volume[0][0])

    volume_name = ''

    for letter in act_name:
        if letter != "_":
            volume_name += letter
        else:
            break

    new_name = f"{volume_name}_dimensions_{x}_{y}_{z}_interpolation_{current_resolution[act_name[0]][2]}nm_to_{target_resolution[act_name[0]][2]}nm.npy"
    return new_name


def resize_volumes_changed_resolution(
    path_to_new_resolution: str,
):
    for arq in os.listdir(path_to_new_resolution):
            path_arq = os.path.join(path_to_new_resolution, arq)
            volume_changed_resolution = np.load(path_arq)
            dim = len(volume_changed_resolution[0][0])
            volume_resized = []
            print(f"Shape do Volume {arq}: {volume_changed_resolution.shape}")
            print(f"Shape da Fatia do Volume {arq}: {volume_changed_resolution[:,:,0].shape}")
            for k in range(dim):

                slice_resized = resize(volume_changed_resolution[:,:,k], (128, 128)) * 255
                volume_resized.append(slice_resized.T)

            
            volume_resized = np.array(volume_resized).astype(np.uint8).T
            new_name = generate_new_name(arq, volume_resized)
            path_to_save = os.path.join(path_new_resolution_resized, new_name)
            print(f"Shape do Volume eh: {volume_resized.shape}. Coords eh {len(volume_resized)},{len(volume_resized[0])},{len(volume_resized[0][0])}")
            np.save(path_to_save, volume_resized)

def resample_volume_to_resolution(volume: np.ndarray, name_volume: str, current_resolution_nm: tuple[float, float, float], target_resolution_nm: tuple[float, float, float]) -> np.ndarray:
    
    print(f"Alterando Resolucao do Volume {name_volume}")
    zoom_factors = []
    dim_resolution = len(current_resolution_nm)
 
    for i in range(dim_resolution):
        zoom_factors.append(current_resolution_nm[i]/target_resolution_nm[i])
    
    zoom_factors = tuple(zoom_factors)  
    resampled_volume = zoom(volume, zoom=zoom_factors, order=1)
    return resampled_volume
