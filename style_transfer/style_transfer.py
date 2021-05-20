"""
Скрипт для переноса стиля на
"""

import numpy as np
import torch
import h5py

from style_transfer.StylesPicker import StylesPicker
from style_transfer.model_AdaIN import WCT2AdaIN


# пути до обученных decoder и encoder
encoder_path = '../input/wct2-coco-trained/encoder_coco.pth'
decoder_path = '../input/wct2-coco-trained/decoder_coco.pth'

# датасет с content изображениями
input_images_path = '../input/covid-dset/COVID_dset.hdf5'

# библиотека стилей
style_lib_path = '../input/mayo-style-and-test/style_data.hdf5'

# на каких слоях нужно делать перенос стиля
transfer_at = {'skip', 'encoder', 'decoder'}

# способ анпулинга
option_unpool = 'cat5'  # 'cat5' или 'sum'


def load_images(path):
    with h5py.File(path, 'r') as hf:
        return np.array(hf.get('LDCT'))


def style_transfer(wct_loaded_model, content_image_np, style_image_np, device):
    content_image = torch.tensor([content_image_np, content_image_np, content_image_np],
                                 dtype=torch.float).to(device)
    style_image = torch.tensor([style_image_np, style_image_np, style_image_np],
                                 dtype=torch.float).to(device)
    with torch.no_grad():
        styled_image = wct_loaded_model.transfer(content_image, style_image, alpha=1)
    numpy_img = styled_image.clamp_(0, 1).cpu()[0, :, :, :].permute(1, 2, 0).numpy()
    return numpy_img[..., 0]


def run_style_transfer():
    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    wct_model = WCT2AdaIN(encoder_path=encoder_path, decoder_path=decoder_path,
                     transfer_at=transfer_at, option_unpool=option_unpool,
                     device=device)
    input_images = load_images(input_images_path)
    style_picker = StylesPicker(style_lib_path)

    images_number, h, w = input_images.shape
    styled_images = np.empty((0, w, w))
    for idx in range(images_number):
        input_image = input_images[idx, ...]
        bf_style = style_picker.find_style(input_image)
        styled_image = style_transfer(wct_model, input_image, bf_style, device, h)
        styled_images = np.concatenate((styled_images, styled_image))

    with h5py.File(input_images_path, 'r+') as hf:
        hf.create_dataset("LD_Styled", data=styled_images)
