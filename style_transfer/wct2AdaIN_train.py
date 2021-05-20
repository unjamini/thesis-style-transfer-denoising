import numpy as np
import torch
import h5py

from style_transfer.model_AdaIN import WCT2AdaIN
from style_transfer.wctutils import Timer


def load_images(path, name='LDCT'):
    with h5py.File(path, 'r') as hf:
        return np.array(hf.get(name))


def main():
    # путь до VGG-19 encoder
    encoder_path = ''
    decoder_path = ''

    # датасет для обучения (MS COCO-2017)
    input_images_path = ''
    train_images = load_images(input_images_path)

    # на каких слоях нужно делать перенос стиля
    transfer_at = {'skip', 'encoder', 'decoder'}

    # способ анпулинга
    option_unpool = 'cat5'  # 'cat5' или 'sum'

    device = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)
    with Timer('Elapsed time in whole WCT: {}', True):
        wct2 = WCT2AdaIN(None, None, transfer_at=transfer_at, option_unpool=option_unpool, device=device, verbose=True)
        wct2.train(train_images, device)
        torch.save(wct2.encoder.state_dict(), encoder_path)
        torch.save(wct2.decoder.state_dict(), decoder_path)
