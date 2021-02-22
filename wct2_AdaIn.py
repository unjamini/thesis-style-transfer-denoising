import numpy as np
import pandas as pd
import csv

import os
import tqdm

import torch
from torchvision.utils import save_image
import h5py
from matplotlib import pyplot as plt


from wct2_style_transfer.model import WaveEncoder, WaveDecoder
from wct2_style_transfer.utils.core import feature_wct
from wct2_style_transfer.utils.io import  Timer, open_image, load_segment, compute_label_info


class WCT2:
    def __init__(self, encoder_path='./encoder_coco_lidc1.pth', decoder_path='./decoder_coco_lidc1.pth',
                 transfer_at=['encoder', 'skip', 'decoder'],
                 option_unpool='cat5', device='cuda:0', verbose=True):

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
            transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)

        self.dec_optim = None
        self.lr = 0.001

        self.recon_weight = 1
        self.feature_weight = 1

        self.encoder.load_state_dict(
            torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(
            torch.load(decoder_path, map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, alpha=1):
        style = style[:, :3, ...]
        content = content[:, :3, ...]
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level], alpha)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component],
                                                                       style_skips[skip_level][component],
                                                                       alpha)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level], alpha)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat

    def train(self, train_data, device, metrics_file='./metrics.csv'):
        output_file = open(metrics_file, 'w')
        with output_file:
            writer = csv.writer(output_file)
            running_loss = 0.0
            dset_len = train_data.shape[0]
            for i in range(dset_len):
                real_image = torch.tensor([[train_data[i, :, :], train_data[i, :, :], train_data[i, :, :]]],
                                          dtype=torch.float).to(device)
                if real_image.shape[1] != 3:
                    print('Image must have 3 chanels')
                    continue
                #  обучение только для весов декодера
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.dec_optim = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, self.decoder.parameters()),
                    lr=self.lr
                )
                self.dec_optim.zero_grad()
                feature, skips = self.encoder.forward(real_image)
                recon_image = self.decoder.forward(feature, skips)
                feature_recon, _ = self.encoder.forward(recon_image)
                mse_loss = torch.nn.MSELoss(size_average=True)
                recon_loss = mse_loss(recon_image, real_image)
                feature_loss = mse_loss(feature_recon, feature.detach())
                loss = self.recon_weight * recon_loss + self.feature_weight * feature_loss
                loss.backward()
                self.dec_optim.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    writer.writerows([[i, 0, recon_loss.item(), running_loss / 200]])
                    running_loss = 0.0


def style_transfer(wct_loaded_model, content_image_path, style_image_path, image_size=512):
    with torch.no_grad():
        img = wct2.transfer(content_image, style_image, content_segment, style_segment, alpha=config['alpha'])