import numpy as np
import torch
import h5py

from style_picker import StylesPicker
from wct2_style_transfer.utils.io import open_image, save_image
from wct2_AdaIn import WCT2

encoder_path = ''
decoder_path = ''

input_images_path = ''
style_lib_path = ''
transfer_at = {'encoder', 'decoder', 'skip'}
option_unpool = 'cat5' # 'cat5' или 'sum'


def load_images(path):
    with h5py.File(path, 'r') as hf:
        return np.array(hf.get('LDCT'))


def style_transfer(wct_loaded_model, content_image_np, style_image_np, image_size=512):
    '''
    :param wct_loaded_model: загруженная модель
    :param content_image_np:
    :param style_image_np:
    :param image_size:
    :return: numpy array стилизованное изображение
    '''
    # исправить тут - должен принимать np.array, а не путь
    content_image = open_image(content_image_np, image_size).to(device)
    style_image = open_image(style_image_np, image_size).to(device)  # и тут
    with torch.no_grad():
        styled_image = wct_loaded_model.transfer(content_image, style_image, alpha=1)
    numpy_img = styled_image.clamp_(0, 1).cpu()[0, :, :, :].permute(1, 2, 0).numpy()
    # plt.imshow(numpy_img)
    return numpy_img[..., 0]  # возможно нужно поменять возвращаемый канал (или суммировать по всем?)
    # save_image(styled_image.clamp_(0, 1), 'styled' + content_image_path, padding=0)
    # for i in range(3):
    #     plt.imsave(str(i) + 'styled' + content_image_path, numpy_img[:, :, i], cmap='gray')


device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
device = torch.device(device)

wct_model = WCT2(encoder_path=encoder_path, decoder_path=decoder_path,
                 transfer_at=transfer_at, option_unpool=option_unpool,
                 device=device)
input_images = load_images(input_images_path)
style_picker = StylesPicker(style_lib_path)

images_number, h, w = input_images.shape
styled_images = np.empty((0, w, w))
for idx in range(images_number):
    input_image = input_images[idx, ...]
    bf_style = style_picker.find_style(input_image)
    styled_image = style_transfer(wct_model, input_image, bf_style, h)
    styled_images = np.concatenate((styled_images, styled_image))

with h5py.File(input_images_path, 'r+') as hf:
    hf.create_dataset("LD_Styled", data=styled_images)
