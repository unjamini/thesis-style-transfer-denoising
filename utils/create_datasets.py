import numpy as np
import os
import h5py
import scipy.ndimage
import matplotlib.pyplot as plt
from pydicom import dcmread

import random

directory = '/Users/evgenia/Desktop/диплом/Mayo-Dataset/'

# path_idxs = [2, 4, 12, 16, 27, 30, 50, 52, 67, 77, 81, 95, 99, 107, 111,
#              120, 121, 124, 128, 135, 158, 160, 162, 166, 170, 179,
#              190, 193, 202, 203, 218, 219, 224, 227, 232, 234, 241, 246,
#              249, 252, 257, 258, 261, 267, 268, 280, 295, 296] # 21, 130 - разное кол-во LD ND

# 295 и 296 оставила для тестирования


def load_scan(path):
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        # distance between slices, finds slice tkickness if not availabe
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    # read the dicom images, find HU numbers (padding, intercept, rescale), and make a 4-D array,

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0

    image[image == padding] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_samples(path):
    scans = load_scan(path)
    return get_pixels_hu(scans)


def extract_patches(ld_image, nd_image, patch_size=40, stride=32, skip_images=40, keep_picked=False):
    images_num, h, w = ld_image.shape
    images_num //= skip_images
    out_ld = np.empty((0, patch_size, patch_size))
    out_nd = np.empty((0, patch_size, patch_size))
    if keep_picked:
        picked = np.empty((0, h, w))
    sz = ld_image.itemsize
    shape = ((h - patch_size) // stride + 1, (w - patch_size) // stride + 1, patch_size, patch_size)
    strides = sz * np.array([w * stride, stride, w, 1])
    images_idxs = random.sample(range(1, ld_image.shape[0]), images_num)
    for d, idx in enumerate(images_idxs):
        ld_patches = np.lib.stride_tricks.as_strided(ld_image[idx, :, :], shape=shape, strides=strides)
        ld_blocks = ld_patches.reshape(-1, patch_size, patch_size)
        out_ld = np.concatenate((out_ld, ld_blocks[:, :, :]))
        if keep_picked:
            picked = np.concatenate((picked, ld_image[idx, :, :].reshape(1, h, w)))
        nd_patches = np.lib.stride_tricks.as_strided(nd_image[idx, :, :], shape=shape, strides=strides)
        nd_blocks = nd_patches.reshape(-1, patch_size, patch_size)
        out_nd = np.concatenate((out_nd, nd_blocks[:, :, :]))
        print(d, end=' ')
    if keep_picked:
        return out_ld[:, :, :], out_nd[:, :, :], picked
    return out_ld[:, :, :], out_nd[:, :, :]


def write_hdf5(low_dose_images, normal_dose_images, filename="train_data.hdf5"):
    with h5py.File(filename, "w") as f:
        f.create_dataset("LDCT", data=low_dose_images)
        f.create_dataset("NDCT", data=normal_dose_images)


def save_style_lib_hdf5(style_images, filename="style_data.hdf5"):
    with h5py.File(filename, "w") as f:
        f.create_dataset("style", data=style_images)


def compile_image(image_slices, image_size=512, slice_size=40, stride=32):
    new_image = np.zeros((image_size, image_size))
    slices_num = image_size // stride
    h = 0
    for i in range(slices_num):
        w = 0
        for j in range(slices_num):
            cur_idx = i * slices_num + j
            for ii in range(slice_size):
                for jj in range(slice_size):
                    if h + ii < image_size and w + jj < image_size and not new_image[h + ii, w + jj]:
                        new_image[h + ii, w + jj] = image_slices[cur_idx][ii, jj]
                    elif h + ii < image_size and w + jj < image_size:
                        new_image[h + ii, w + jj] = (new_image[h + ii, w + jj] + image_slices[cur_idx][ii, jj]) / 2
            w += stride
        h += stride
    return new_image

    
if __name__ == '__main__':
    patch_size = 40
    ld_data = np.empty((0, patch_size, patch_size))
    nd_data = np.empty((0, patch_size, patch_size))
    picked_data = np.empty((0, 512, 512))
    for patient in os.listdir(directory):
        if patient != '.DS_Store' and patient != 'LDCT-and-Projection-data':
            path_low = os.path.join(directory, patient, 'Low')
            path_full = os.path.join(directory, patient, 'Full')
            pat_low = load_scan(path_low)
            pat_full = load_scan(path_full)
            pixels_low = get_pixels_hu(pat_low)
            pixels_full = get_pixels_hu(pat_full)
            ld_patches, nd_patches, picked_images = extract_patches(pixels_low, pixels_full, keep_picked=True)
            ld_data = np.concatenate((ld_data, ld_patches))
            nd_data = np.concatenate((nd_data, nd_patches))
            picked_data = np.concatenate((picked_data, picked_images))
            print(ld_data.shape)
            print(nd_data.shape)
            print(picked_images.shape)
            print(picked_data.shape)
    write_hdf5(ld_data, nd_data)
    save_style_lib_hdf5(picked_data)
