import numpy as np
from preprocess_CT_image import load_samples, windowing1
from skimage.feature import local_binary_pattern

path_style = '/Users/evgenia/Desktop/диплом/Mayo-Dataset/2/Low/'

style_images = []
style_mean_arr = []
style_std_arr = []
style_lbp_arr = []
style_hist_arr = []


def get_corr_with_image(image_hist, style_idx):
    return np.correlate(image_hist, style_hist_arr[style_idx])


def get_lbp(image):
    return local_binary_pattern(image, 24, 3, 'ror')


def get_img_mean(image):
    return np.mean(image)


def get_img_std(image):
    return np.std(image)


def compare_func(image_mean, image_std, style_idx):
    return np.sqrt((image_mean - style_mean_arr[style_idx])**2) +\
           np.sqrt((image_std - style_std_arr[style_idx])**2)


def load_style_images(style_images_path):
    style_pixels = windowing1(load_samples(style_images_path))
    style_images.append(style_pixels)
    for idx in range(len(style_images)):
        style_mean_arr.append(get_img_mean(style_images[idx]))
        style_std_arr.append(get_img_std(style_images[idx]))
        lbp = get_lbp(style_images[idx])
        style_lbp_arr.append(lbp)
        hist, bins = np.histogram(lbp.ravel(), 256, [0, 256])
        style_hist_arr.append(hist)


def find_style_image(input_image):
    image_lbp = get_lbp(input_image)
    image_std = get_img_std(input_image)
    image_mean = get_img_mean(input_image)
    image_hist = np.histogram(image_lbp.ravel(), 256, [0, 256])
    corr_coeff_arr = []
    for idx in range(len(style_images)):
        corr_coeff_arr.append(get_corr_with_image(image_hist, idx))
    corr_coeff_arr = np.array(corr_coeff_arr)
    top_10_idx = (-corr_coeff_arr).argsort()[:10]
    top_10_styles = style_images[top_10_idx]
    min_val = 1e10
    res_idx = -1
    for idx in top_10_idx:
        value = compare_func(image_mean, image_std, idx)
        if value < min_val:
            res_idx = idx
            min_val = value
    return res_idx

# print(res_idx)