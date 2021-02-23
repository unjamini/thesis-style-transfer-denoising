import numpy as np
import h5py
import matplotlib.pyplot as plt

from train_cnn_vgg import create_model, perceptual_loss
from utils.create_datasets import compile_image, extract_patches

model_path = './res/Weights/weights_DRL_edge4d_adam2_perceptual70_mse30_pig.h5'


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
    patches_num, h, w = patches.shape
    img_num = patches_num // patches_per_image
    out = np.empty((0, image_shape, image_shape))
    for i in range(img_num):
        # перепроверить тут размерность !!!
        res = compile_image(np.sum(patches[i*patches_per_image:(i+1)*patches_per_image, ...], axis=-1))
        out = np.concatenate((out, res))
    return out


def plotting(input, label, result):
    fig, axs = plt.subplots(1, 3)
    axs[1].imshow(result, cmap='gray')
    axs[1].set_title('Результат')
    axs[0].imshow(input, cmap='gray')
    axs[0].set_title('LDCT')
    axs[2].imshow(label, cmap='gray')
    axs[2].set_title('NDCT')
    plt.show()


def main():
    model_edge_p_mse = create_model()
    model_edge_p_mse.load_weights(model_path)

    data, styled_data, labels = load_and_prepare_data('./test_data.hdf5')
    data_patches = extract_patches(data)
    styled_patches = extract_patches(styled_data)
    [pred_labels, pred_labels] = model_edge_p_mse.predict(data_patches, batch_size=8, verbose=1)
    [pred_labels_styled, pred_labels_styled] = model_edge_p_mse.predict(styled_patches, batch_size=8, verbose=1)

    pred_images = compile_all(pred_labels)
    pred_images_styled = compile_all(pred_labels_styled)

    # добавить тут сравнение по метрикам
    # SSIM, PSNR + perceptual_loss
    # сравнивать с labels
