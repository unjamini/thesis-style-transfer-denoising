import numpy as np
import os
import h5py
from pydicom import dcmread

import random

directory = '../../Mayo-Dataset/'


def load_scan(path):
    # загрузка dicom-изображения
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices


def get_pixels_hu(slices):
    '''

    :param slices: Загруженное изображение в dicom формате
    :return: Конвертированное изображение (ndarray)
    '''
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    image[image == padding] = 0
    # конвертирование в Hounsfield units (HU)
    for idx, slice in enumerate(slices):
        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope

        if slope != 1:
            image[idx] = slope * image[idx].astype(np.float64)
            image[idx] = image[idx].astype(np.int16)
        image[idx] += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def load_samples(path):
    '''

    :param path: Путь к dicom изображению
    :return: ndarray изображение
    '''
    scans = load_scan(path)
    return get_pixels_hu(scans)


def compile_image(image_slices, image_size=512, slice_size=40, stride=32):
    '''

    Функнция для восстановления полного изображения из частей

    :param image_slices: массив частей изображения
    :param image_size: размер изображения
    :param slice_size: размеры слайсов
    :param stride: отступ для нарезки на слайсы
    :return: восстановленное изображение
    '''
    new_image = np.zeros((image_size, image_size, 3))
    slices_num = image_size // stride
    h = 0
    for i in range(slices_num):
        w = 0
        for j in range(slices_num):
            cur_idx = i * slices_num + j
            for ii in range(6, slice_size - 2):
                for jj in range(2, slice_size - 6):
                    if h + ii < image_size and w + jj < image_size and not new_image[h + ii, w + jj, 0]:
                        new_image[h + ii, w + jj] = image_slices[cur_idx][ii, jj]
            w += stride
        h += stride
    return new_image


def extract_patches(image, patch_size=40, stride=32):
    """
    Функция для нарезки изображения на патчи размера (patch_size, patch_size)

    :param image: Изображение низкого качества
    :param patch_size: Размер патча для нарезки
    :param stride: Размер отступа для нарезки
    """
    images_num, h, w = image.shape
    out = np.empty((0, patch_size, patch_size))
    sz = image.itemsize
    shape = (h // stride, w // stride, patch_size, patch_size)
    strides = sz * np.array([w * stride, stride, w, 1])
    images_idxs = list(range(images_num))
    for i in images_idxs:
        patches = np.lib.stride_tricks.as_strided(image[i, ...], shape=shape, strides=strides)
        blocks = patches.reshape(-1, patch_size, patch_size)
        out = np.concatenate((out, blocks))
    return out


def extract_patches_training(ld_image, nd_image, patch_size=40, stride=32, skip_images=40, keep_picked=False):
    '''

    Функция для нарезки пар изображений на патчи размера (patch_size, patch_size)

    :param ld_image: Изображение низкого качества
    :param nd_image: Изображение высокого качества
    :param patch_size: Размер патча для нарезки
    :param stride: Размер отступа для нарезки
    :param skip_images: Сколько последовательных слайсов для одного пациента пропускать
    :param keep_picked: Нужно ли сохранять изображения для базы стилей
    :return:
    '''
    images_num, h, w = ld_image.shape
    images_num //= skip_images if skip_images else 1
    out_ld = np.empty((0, patch_size, patch_size))
    out_nd = np.empty((0, patch_size, patch_size))
    picked = np.empty((0, h, w))
    sz = ld_image.itemsize
    shape = ((h - patch_size) // stride + 2, (w - patch_size) // stride + 2, patch_size, patch_size)
    strides = sz * np.array([w * stride, stride, w, 1])
    images_idxs = random.sample(range(1, ld_image.shape[0]), images_num)
    for d, idx in enumerate(images_idxs):
        ld_patches = np.lib.stride_tricks.as_strided(ld_image[idx, ...], shape=shape, strides=strides)
        ld_blocks = ld_patches.reshape(-1, patch_size, patch_size)
        out_ld = np.concatenate((out_ld, ld_blocks))
        if keep_picked:
            picked = np.concatenate((picked, ld_image[idx, ...].reshape(1, h, w)))
        nd_patches = np.lib.stride_tricks.as_strided(nd_image[idx, ...], shape=shape, strides=strides)
        nd_blocks = nd_patches.reshape(-1, patch_size, patch_size)
        out_nd = np.concatenate((out_nd, nd_blocks))
    if keep_picked:
        return out_ld, out_nd, picked
    return out_ld, out_nd


def write_hdf5(low_dose_images, normal_dose_images, filename="train_data.hdf5"):
    '''
    Сохранение датасета для обучения в hdf5 формате

    :param low_dose_images: массив слайсов изображений низкого качества
    :param normal_dose_images: массив слайсов изображений высокого качества
    :param filename: имя файла для сохранения

    '''
    with h5py.File(filename, "w") as f:
        f.create_dataset("LDCT", data=low_dose_images)
        f.create_dataset("NDCT", data=normal_dose_images)


def save_style_lib_hdf5(style_images, filename="style_data.hdf5"):
    '''
    Функция для создания базы стилей для переноса из LD изображений обучающего датасета

    :param style_images: массив изображений низкого качества
    :param filename: имя файла для сохранения
    :return:
    '''
    with h5py.File(filename, "w") as f:
        f.create_dataset("style", data=style_images)

    
def create_dset_for_trainig():
    patch_size = 40
    ld_data = np.empty((0, patch_size, patch_size))
    nd_data = np.empty((0, patch_size, patch_size))
    picked_data = np.empty((0, 512, 512))
    for patient in os.listdir(directory):
        if patient != '.DS_Store' and patient != 'LDCT-and-Projection-data':
            path_low = os.path.join(directory, patient, 'Low')
            path_full = os.path.join(directory, patient, 'Full')
            pixels_low = load_samples(path_low)
            pixels_full = load_samples(path_full)

            ld_patches, nd_patches, picked_images = extract_patches_training(pixels_low, pixels_full, keep_picked=True)
            ld_data = np.concatenate((ld_data, ld_patches))
            nd_data = np.concatenate((nd_data, nd_patches))
            picked_data = np.concatenate((picked_data, picked_images))
    write_hdf5(ld_data, nd_data)
    save_style_lib_hdf5(picked_data)
