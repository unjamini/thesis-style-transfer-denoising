import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
# from preprocess_CT_image import load_samples, windowing1
from skimage.feature import local_binary_pattern


def get_lbp(image):
    return local_binary_pattern(np.copy(image), 24, 3, 'ror')


def get_hist(lbp):
    image_hist, _ = np.histogram(lbp.ravel(), 256, [0, 256])
    return image_hist


def get_corr_with_image(image_hist, style_hist):
    return np.correlate(image_hist, style_hist)


def get_img_mean(image):
    return np.mean(image)


def get_img_std(image):
    return np.std(image)


class StylesPicker:
    def __init__(self, styles_hdf5_filename):
        self.style_lib = None
        self.style_lib_filename = styles_hdf5_filename
        self.style_mean_arr = None
        self.style_std_arr = None
        self.style_lbp_arr = None
        self.style_hist_arr = None
        pass

    def compare_func(self, image_mean, image_std, style_idx):
        return np.sqrt((image_mean - self.style_mean_arr[style_idx]) ** 2) + \
               np.sqrt((image_std - self.style_std_arr[style_idx]) ** 2)

    def initialize(self):
        if self.style_lib is None:
            return
        self.style_mean_arr = np.mean(self.style_lib, axis=(1, 2))
        self.style_std_arr = np.std(self.style_lib, axis=(1, 2))
        print(self.style_lib.shape, self.style_mean_arr.shape)
        self.style_lbp_arr = np.vectorize(get_lbp, signature='(n,m)->(n,n)')(self.style_lib)
        print(self.style_lbp_arr.shape)
        self.style_hist_arr = np.vectorize(get_hist, signature='(n,m)->(256)')(self.style_lbp_arr)
        print(self.style_lib.shape, self.style_mean_arr.shape, self.style_lbp_arr.shape, self.style_hist_arr.shape)

    def find_style(self, image):
        if self.style_lib is None:
            with h5py.File(self.style_lib_filename, 'r') as hf:
                self.style_lib = np.array(hf.get('style'))
            self.initialize()
        img_mean = np.mean(image)
        img_std = np.std(image)
        image_lbp = get_lbp(image)
        image_hist, _ = np.histogram(image_lbp.ravel(), 256, [0, 256])
        func = lambda x: get_corr_with_image(image_hist, x)
        corr_coef_with_styles = np.apply_along_axis(func, 1, self.style_hist_arr)
        print(corr_coef_with_styles.shape)
        print(corr_coef_with_styles)
        top_10_idx = (-corr_coef_with_styles).argsort()[:10]
        func2 = lambda x: self.compare_func(img_mean, img_std, x)
        compare_values = np.vectorize(func2)(top_10_idx)
        res_idx = top_10_idx[np.argmin(compare_values)]
        return self.style_lib[res_idx, :, :]


# s_lib = StylesPicker('/Users/evgenia/Desktop/диплом/cnn-vgg/style_data.hdf5')
# s = Image.open('in.png')
# data = np.asarray(s)[..., 0]
# result = s_lib.find_style(data)
# plt.imshow(result[0, ...])
# plt.show()
