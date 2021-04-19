"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0
"""
import torch
import datetime


"""
Функции для Whitening&Coloring трансформаций. Эти трансформации заменены на AdaIN, но WCT использовалось
для сравнения работы различных методов переноса стиля.

"""


def svd(feat, iden=False, device='cpu'):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    if iden:
        conv += torch.eye(size[0]).to(device)
    u, e, v = torch.svd(conv, some=False)
    return u, e, v


def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim, eps=0.00001):
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r


def wct_core(cont_feat, styl_feat, weight=1, registers=None, device='cpu'):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    if not registers:
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

        styl_feat = get_squeeze_feat(styl_feat)
        s_mean = torch.mean(styl_feat, 1)
        _, s_e, s_v = svd(styl_feat, iden=True, device=device)
        k_s = get_rank(s_e, styl_feat.size()[0])
        s_d = (s_e[0:k_s]).pow(0.5)
        EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d) * weight), (s_v[:, 0:k_s].t()))

        if registers is not None:
            registers['EDE'] = EDE
            registers['s_mean'] = s_mean
            registers['c_v'] = c_v
            registers['c_e'] = c_e
    else:
        EDE = registers['EDE']
        s_mean = registers['s_mean']
        _, c_e, c_v = svd(cont_feat, iden=True, device=device)

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    targetFeature = torch.mm(EDE, whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


"""
Функции для осуществления Adaptive Instance Normalization.
"""


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """
    Операция AdaIN
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def feature_wct(content_feat, style_feat, alpha=1):
    """
    Основная функциия для переноса стиля
    :param content_feat: Выделенные HH, LH, HL признаки из изображения с контентом
    :param style_feat: Выделенные HH, LH, HL признаки из изображения со стилем
    :param alpha: Параметр смешивания
    :return:
    """
    target_feature = adaptive_instance_normalization(content_feat, style_feat)  # ada-in
    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


class Timer:
    def __init__(self, msg='Elapsed time: {}', verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))
