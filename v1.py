# Copyright (c) smarsu. All Rights Reserved.

"""Run disturb image with transfer style."""

import os.path as osp
import csv
import cv2
import numpy as np
import torch
import torchvision
# from torchvision.models.utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def resnet50():
    """Load resnet50"""
    return torchvision.models.resnet.resnet50(pretrained=True, progress=True)


def vgg16():
    """Load vgg16"""
    return torchvision.models.vgg.vgg16(pretrained=True, progress=True)


def mobilenet():
    """Load mobilenet"""
    return torchvision.models.mobilenet.mobilenet_v2(pretrained=True, progress=True)


class MobileNetV2Custom(torchvision.models.mobilenet.MobileNetV2):
    def _forward_impl(self, x, idx):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        feat = x
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x, self._get_feature_map(feat, idx)

    
    def _get_feature_map(self, feat, idx):
        feat = feat.permute(0, 2, 3, 1)
        feat = self.classifier(feat)
        feat = feat.reshape(-1, 1000)
        feat = feat[:, idx]
        feat = torch.argmin(feat, -1)
        return feat


    def forward(self, x, idx):
        return self._forward_impl(x, idx)


def mobilenet_v2_custom(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2Custom(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def load_dev(path):
    """"""
    with open(path, "r") as fb:
        reader = list(csv.reader(fb))[1:]  # no need for title
        return reader


def center_crop(image, size=(224, 224)):
    """"""
    # assert image.shape == (299, 299, 3)
    # image = cv2.resize(image, (256, 256))
    # h, w, _ = image.shape
    # top = (h - size[0]) // 2
    # bottom = (h - size[0]) - top
    # left = (w - size[1]) // 2
    # right = (w - size[1]) - left

    # image = image[top:h-bottom, left:w-right, :]
    # assert image.shape == (224, 224, 3)

    image = cv2.resize(image, (224, 224))
    return image


def process_image(img_path):
    """ process_image """
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img = cv2.imread(img_path)

    img = center_crop(img)
    img = img[:, :, ::-1]

    img = img.astype(np.float32).transpose(2, 0, 1) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img[np.newaxis, ...]


def norm(image):
    """"""
    image -= np.mean(image, axis=-1, keepdims=True)
    image /= np.sqrt(np.var(image, axis=-1, keepdims=True) + 1e-6)
    return image


def disturb_image(image, idx):
    """"""
    image = image.astype(np.float32)
    image = cv2.resize(image, (224, 224))

    h = idx // 7
    w = idx % 7

    patch = image[h*32:(h+1)*32, w*32:(w+1)*32, :].copy()
    patch = norm(patch)
    patch *= 30
    patch = np.clip(patch, -30, 30)
    # patch -= np.mean(patch, axis=-1, keepdims=True)
    # patch = 30 * patch / np.max(np.abs(patch))
    print(patch)

    image = image.reshape(7, 32, 7, 32, 3)
    image = image.transpose(0, 2, 1, 3, 4)
    image += patch
    image = image.transpose(0, 2, 1, 3, 4)
    image = image.reshape(224, 224, 3)

    image = cv2.resize(image, (299, 299))
    return image


def run_model(model):
    """"""
    lines = load_dev('dev.csv')
    size = len(lines)
    disturb = 0
    fake = 0
    for id, true, target in lines:
        if id != '0c7ac4a8c9dfa802.png':
            continue

        true = int(true)
        target = int(target)

        image = process_image(osp.join('images', id))

        x, feat = mobilenet_model(torch.from_numpy(image).cpu(), true)
        x = x.cpu().detach().numpy()
        feat = feat.cpu().detach().numpy()
        
        disturb += true != np.argmax(x) + 1
        fake += target == np.argmax(x) + 1
        print(true, np.argmax(x) + 1, true == np.argmax(x) + 1, target, target == np.argmax(x) + 1, feat)
        print(disturb, fake)

        image = disturb_image(cv2.imread(osp.join('images', id)), feat)
        cv2.imwrite(osp.join('fake_images', id), image)


if __name__ == '__main__':
    # resnet50_model = resnet50().eval()
    # vgg16_model = vgg16().eval()
    # mobilenet_model = mobilenet().eval()

    mobilenet_model = mobilenet_v2_custom(pretrained=True, progress=True).eval().cpu()

    run_model(mobilenet_model)
