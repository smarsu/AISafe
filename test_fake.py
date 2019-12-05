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

import resnet
import mobilenet


def resnet50():
    """Load resnet50"""
    return resnet.resnet50(pretrained=True, progress=True)


def resnet152():
    """Load resnet50"""
    return resnet.resnet152(pretrained=True, progress=True)


def vgg16():
    """Load vgg16"""
    return torchvision.models.vgg.vgg16(pretrained=True, progress=True)


def mobilenetv2():
    """Load mobilenet"""
    return mobilenet.mobilenet_v2(pretrained=True, progress=True)


def densenet121():
    """"""
    return torchvision.models.densenet.densenet121(pretrained=True, progress=True)


class MobileNetV2Custom(mobilenet.MobileNetV2):
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
    """Convert to center crop"""
    assert image.shape == (299, 299, 3)
    image = cv2.resize(image, (256, 256))
    h, w, _ = image.shape
    top = (h - size[0]) // 2
    bottom = (h - size[0]) - top
    left = (w - size[1]) // 2
    right = (w - size[1]) - left

    image = image[top:h-bottom, left:w-right, :]
    assert image.shape == (224, 224, 3)

    # image = cv2.resize(image, (224, 224))
    return image


def process_image(img):
    """ process_image """
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

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


def check_distance(src_path, target_path):
    x = cv2.imread(target_path).astype(np.float32)
    y = cv2.imread(src_path).astype(np.float32)
    print(np.max(np.abs(x - y)))


def clip(x, target):
    x = x.astype(np.int32)

    large = x + 32
    large = np.minimum(large, 255).astype(np.float32)

    small = x - 32
    small = np.maximum(small, 0).astype(np.float32)

    src_target = target = target.astype(np.float32)
    target = np.clip(target, small, large)

    sum = np.sum(src_target - target)
    if sum > 0:
        print("clip it")
    return target


def run_model(model, is_fake=True):
    """"""
    lines = load_dev('dev.csv')
    size = len(lines)
    disturb = 0
    fake = 0
    for id, true, target in lines:
        # if id != '0c7ac4a8c9dfa802.png':
        #     continue
        if id != '137ab6ca314e9e35.png':
            continue
        src_path = osp.join('images', id)
        fake_path = osp.join('fake_images', id) if is_fake else osp.join('images', id)

        true = int(true)
        target = int(target)

        check_distance(src_path, fake_path)
        image = clip(cv2.imread(src_path), cv2.imread(fake_path))
        image = process_image(image)

        x = model(torch.from_numpy(image).cuda())
        x = x.cpu().detach().numpy()
        
        disturb += true != np.argmax(x) + 1
        fake += target == np.argmax(x) + 1
        print(true, np.argmax(x) + 1, true == np.argmax(x) + 1, target, target == np.argmax(x) + 1)
        print(disturb, fake)


if __name__ == '__main__':
    # resnet50_model = resnet50().eval()
    resnet152_model = resnet152().eval().cuda()
    # vgg16_model = vgg16().eval()
    # mobilenet_model = mobilenetv2().eval()
    # densenet121_model = densenet121().eval()

    run_model(resnet152_model, is_fake=True)
