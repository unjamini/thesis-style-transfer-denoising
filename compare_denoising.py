import numpy as np
import h5py

from denoiser.train_cnn_vgg import create_model
from utils.create_datasets import compile_image, extract_patches
from utils.comparing_utils import get_metrics_lists ,boxplot


model_path = './training-res/Weights/weights_DRL_edge4d_adam2_perceptual70_mse30_pig.h5'


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('LDCT'))
        styled = np.array(hf.get('LD_Styled'))
        labels = np.array(hf.get('NDCT'))
        return data, styled, labels


def load_and_prepare_data(file):
    data, styled_data, labels = read_hdf5(file)
    data = (data[:, :, :, None] / 4095).astype(np.float32)
    styled_data = (styled_data[:, :, :, None] / 4095).astype(np.float32)
    labels = (labels[:, :, :, None] / 4095).astype(np.float32)
    return data, styled_data, labels


def compile_all(patches, patches_per_image=256, image_shape=512):
    patches_num, h, w, c = patches.shape
    img_num = patches_num // patches_per_image
    out = np.empty((0, image_shape, image_shape, 3))
    for i in range(img_num):
        res = compile_image(patches[i*patches_per_image:(i+1)*patches_per_image, ...])
        out = np.concatenate((out, res[None, :, :, :]))
    return out


def main():
    # загрузка весов модели
    model_edge_p_mse = create_model()
    model_edge_p_mse.load_weights(model_path)

    # загрузка и нарезка на патчи
    data, styled_data, labels = load_and_prepare_data('./test_data.hdf5')
    data_patches = extract_patches(data)
    styled_patches = extract_patches(styled_data)

    # удаление шума для исходных изображений и для стилизованных
    [_, pred_labels] = model_edge_p_mse.predict(data_patches, batch_size=8, verbose=1)
    [_, pred_labels_styled] = model_edge_p_mse.predict(styled_patches, batch_size=8, verbose=1)

    # восстановление картинок из патчей
    pred_images = compile_all(pred_labels)
    pred_images_styled = compile_all(pred_labels_styled)

    # подсчёт метрик, построение боксплотов для них
    psnr_res, psnr_res_styled, ss_res, ss_res_styled = get_metrics_lists(data_patches, pred_labels, pred_labels_styled)
    boxplot(psnr_res, psnr_res_styled, 'PSNR',  'без слоёв пропуска')
    boxplot(ss_res, ss_res_styled, 'SSIM',  'без слоёв пропуска')

