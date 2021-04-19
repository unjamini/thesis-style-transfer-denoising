import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def get_psnr(image, noisy):
    """
    Функция для подсчёта PSNR между двумя изображениями

    :param image: Изображение высокого качества
    :param noisy: Зашумлённое изображение
    :return:
    """
    diff = image - noisy
    diff = diff.flatten('C')
    rmse = np.sqrt(np.mean(diff ** 2.))
    return 20 * np.log10(np.max(image) / rmse)


def get_ssim(image, noisy):
    """
    Функция для подсчёта SSIM между двумя изображениями

    :param image: Изображение высокого качества
    :param noisy: Зашумлённое изображение
    :return:
    """
    ssim0 = 0
    for i in range(image.shape[0]):
        ssim0 += ssim(image[i, :, :, 0], noisy[i, :, :, 0])
    return ssim0 / image.shape[0]


def get_metrics_lists(data_patches, pred_labels, pred_labels_styled, patches_n=256):
    psnr_res = []
    psnr_res_styled = []
    ss_res = []
    ss_res_styled = []

    for i in range(len(data_patches) // patches_n):
        ss_res.append(
            get_ssim(pred_labels[i * patches_n:(i + 1) * patches_n, ...],
                     data_patches[i * patches_n: (i + 1) * patches_n, ...]))
        ss_res_styled.append(
            get_ssim(pred_labels_styled[i * patches_n:(i + 1) * patches_n, ...],
                     data_patches[i * patches_n: (i + 1) * patches_n, ...]))
        psnr_res.append(
            get_psnr(pred_labels[i * patches_n:(i + 1) * patches_n, ...],
                     data_patches[i * patches_n:(i + 1) * patches_n, ...]))
        psnr_res_styled.append(
            get_psnr(pred_labels_styled[i * patches_n:(i + 1) * patches_n, ...],
                     data_patches[i * patches_n:(i + 1) * patches_n, ...]))
    return psnr_res, psnr_res_styled, ss_res, ss_res_styled


def plot_denoising_result_with_label(input, label, result):
    """
    Вывод изображений с результатами денойзинга в случае, если есть ground-truth изображение (label)

    """
    fig, axs = plt.subplots(1, 3)
    for axss in axs.ravel():
        axss.axis('off')
    axs[1].imshow(result, cmap='gray')
    axs[1].set_title('Результат')
    axs[0].imshow(input, cmap='gray')
    axs[0].set_title('LDCT')
    axs[2].imshow(label, cmap='gray')
    axs[2].set_title('NDCT')
    plt.show()


def plot_patches_result(data_patches, pred_labels,
                        pred_labels_styled_full, pred_labels_styled_no_skips,
                        image_n=0, patch_shift=49):
    """
    Вывод изображений для сравненя результатов денойзинга с приближенным патчем.
    Выводится изображение номер image_n и его патч patch_shift.
    """
    fig, axs = plt.subplots(2, 2)
    for axss in axs.ravel():
        axss.axis('off')
    axs[0, 0].imshow(data_patches[image_n + patch_shift], cmap='gray')
    axs[0, 0].set_title(f'Input (LIDC-IDRI)')

    axs[0, 1].imshow(pred_labels[256 * image_n + 49], cmap='gray')
    axs[0, 1].set_title('Result')

    axs[1, 0].imshow(pred_labels_styled_full[256 * image_n + patch_shift], cmap='gray')
    axs[1, 0].set_title('Styled Result (full)')

    axs[1, 1].imshow(pred_labels_styled_no_skips[256 * image_n + patch_shift], cmap='gray')
    axs[1, 1].set_title('Styled Result (no skips)')
    plt.show()


def boxplot(res, res2, metric_name, styling_type=''):
    """
    Построение боксплотов для сранвнения метрики
    """
    plt.rcParams['figure.figsize'] = [10, 12]
    fig, ax = plt.subplots()
    ax.set_title(f'Боксплоты для значений {metric_name}')
    ax.boxplot([res, res2],
                labels=['Без стилизации', f'Стилизация {styling_type}'])

    plt.show()