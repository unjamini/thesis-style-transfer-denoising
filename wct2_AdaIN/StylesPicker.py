import numpy as np
import h5py
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
    """
    Класс для подбора стиля для изображения на входе.
    Изображения подбираются из базы стилей, получаемой на входе.
    Подобор происходит с помощью сравнения гистограм LBP, среднего и дисперсии

    """
    def __init__(self, styles_hdf5_filename):
        """

        :param styles_hdf5_filename: Путь до библиотеки стилей в hdf5 формате
        """
        self.style_lib = None
        self.style_lib_filename = styles_hdf5_filename
        self.style_mean_arr = None
        self.style_std_arr = None
        self.style_lbp_arr = None
        self.style_hist_arr = None
        pass

    def compare_func(self, image_mean, image_std, style_idx):
        return np.sqrt(np.square(image_mean - self.style_mean_arr[style_idx])) + \
          np.sqrt(np.square(image_std - self.style_std_arr[style_idx]))

    def initialize(self):
        if self.style_lib is None:
            return
        self.style_mean_arr = np.mean(self.style_lib, axis=(1, 2))
        self.style_std_arr = np.std(self.style_lib, axis=(1, 2))
        self.style_lbp_arr = np.vectorize(get_lbp, signature='(n,m)->(n,n)')(self.style_lib)
        self.style_hist_arr = np.vectorize(get_hist, signature='(n,m)->(256)')(self.style_lbp_arr)
        print(self.style_lib.shape, self.style_mean_arr.shape, self.style_lbp_arr.shape, self.style_hist_arr.shape)

    def find_style(self, image):
        # используется ленивая инициализация - обработка стилей происходит при обработке первого изображения
        if self.style_lib is None:
            with h5py.File(self.style_lib_filename, 'r') as hf:
                self.style_lib = np.array(hf.get('style'))
            self.initialize()

        # подсчёт среднего, стандартного отклонения и LBP для изображения не входе
        img_mean = np.mean(image)
        img_std = np.std(image)
        image_lbp = get_lbp(image)
        image_hist, _ = np.histogram(image_lbp.ravel(), 256, [0, 256])
        # сравнение гистограмм по их корреляции
        corr_with_styles = np.apply_along_axis(
            lambda x: get_corr_with_image(image_hist, x), 1,
            self.style_hist_arr
        )
        # выбор 10 стилей, по корреляции гистограмм
        top10_idx = (-corr_with_styles[..., 0]).argsort()[:10]
        compare_mean_std = np.vectorize(
            lambda x: self.compare_func(img_mean, img_std, x))(top10_idx)
        # поиск индекса изображения, наиболее близкого по среднему и стандартному отклонению
        res_idx = top10_idx[np.argmin(compare_mean_std)]
        return self.style_lib[res_idx, ...]
