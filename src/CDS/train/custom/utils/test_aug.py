import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
import random


def gen_gaussball(points, radius, img, seg):
    heart_mask = (seg == 1).astype('uint8')
    shape = heart_mask.shape
    gaus_map = np.zeros_like(img)
    # gaus_counter = np.zeros_like(img)
    for idx, point in enumerate(points):
        zmin, ymin, xmin = [max(0, point[i] - 10 * radius[idx]) for i in range(3)]
        zmax, ymax, xmax = [min(shape[i], point[i] + 10 * radius[idx]) for i in range(3)]
        temp_shape = np.array((zmax - zmin, ymax - ymin, xmax - xmin))
        crop_patch = np.zeros(temp_shape)
        crop_patch[temp_shape[0] // 2, temp_shape[1] // 2, temp_shape[2] // 2] = 1
        crop_gaus_map = gaussian_filter(crop_patch, radius[idx])
        gaus_map[zmin: zmax, ymin: ymax, xmin: xmax] += crop_gaus_map
        # gaus_counter[zmin: zmax, ymin: ymax, xmin: xmax] += 1
    
    # gaus_map /= (gaus_counter + 1e-8)
    gaus_map = (gaus_map - np.min(gaus_map)) / (np.max(gaus_map) - np.min(gaus_map) + 1e-8)
    gaus_map_heart = gaus_map * heart_mask
    gaus_map_outheart = gaus_map * (1 - heart_mask)
    
    img_mean = np.sum(img * heart_mask) / (np.sum(heart_mask) + 1e-8)
    img = img * (1 - gaus_map) + gaus_map_heart * img_mean * 0.08 + gaus_map_outheart * img_mean * 0.5
    img = np.clip(img, 0, 1)
    return img


def window_array(vol, win_level=40, win_width=350):
    win = [
        win_level - win_width / 2,
        win_level + win_width / 2,
    ]
    vol = np.clip(vol, win[0], win[1])
    vol -= win[0]
    vol /= win_width
    return vol


if __name__ == "__main__":
    save_path = '../debug'
    os.makedirs(save_path, exist_ok=True)

    processed_data = '../processed_data_1/new_heart_dataset_train_088.npz'

    # 造边缘高斯球
    data = np.load(processed_data)
    for i in data:
        print(i)
    img = data['dcm']
    img_norm = window_array(img)
    seg = data['seg']
    heart_circle = data['heart_circle']
    points = np.argwhere((heart_circle * (seg == 1)) == 1)
    select_points = np.array(random.choices(points, k=random.choice(range(5, 10))))

    radius = random.choices(range(6, 7), k=len(select_points))
    img = gen_gaussball(select_points, radius, img_norm, seg)

    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[img != img_norm] = 1

    sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(save_path, 'a.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(mask), os.path.join(save_path, 'b.nii.gz'))
