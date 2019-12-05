# Copyright (c) smarsu. All Rights Reserved.

"""Run disturb image with transfer style."""

import os.path as osp
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
# # from torchvision.models.utils import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import resnet
import mobilenet


def resnet50():
    """Load resnet50"""
    return torchvision.models.resnet.resnet50(pretrained=True, progress=True)


def resnet152():
    """Load resnet50"""
    return resnet.resnet152(pretrained=True, progress=True)


def vgg16():
    """Load vgg16"""
    return torchvision.models.vgg.vgg16(pretrained=True, progress=True)


def mobilenetv2():
    """Load mobilenet"""
    return mobilenet.mobilenet_v2(pretrained=True, progress=True)


def load_dev(path):
    """"""
    with open(path, "r") as fb:
        reader = list(csv.reader(fb))[1:]  # no need for title
        return reader


def center_crop_inv(image, size=(224, 224)):
    """"""
    # image = cv2.resize(image, (299, 299))
    return image


def center_crop(image, size=(224, 224)):
    """"""
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


def process_image_inv(img, src_img, size=(224, 224)):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img = img.astype(np.float32)
    img = img[0]

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img *= img_std
    img += img_mean
    img *= 255

    # no need

    img = np.transpose(img.astype(np.float32), (1, 2, 0))

    img = img[:, :, ::-1]
    img = center_crop_inv(img)

    src_img = cv2.resize(src_img, (256, 256))
    h, w, _ = src_img.shape
    top = (h - size[0]) // 2
    bottom = (h - size[0]) - top
    left = (w - size[1]) // 2
    right = (w - size[1]) - left
    src_img[top:-bottom, left:-right, :] = img
    src_img = cv2.resize(src_img, (299, 299))
    img = src_img

    return img


def process_image(img):
    """ process_image """
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img = center_crop(img)
    img = img[:, :, ::-1]

    img = img.astype(np.float32).transpose(2, 0, 1)

    # return img

    img /= 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img[np.newaxis, ...]


def clip(x, target):
    x = x.astype(np.float32)

    large = x + 32
    large = np.minimum(large, 255).astype(np.float32)

    small = x - 32
    small = np.maximum(small, 0).astype(np.float32)

    target = target.astype(np.float32)
    target = np.clip(target, small, large)
    return target


def distance(src, target):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img = target
    img = img.astype(np.float32)
    img = img[0]

    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img *= img_std
    img += img_mean
    img *= 255

    # no need

    img = np.transpose(img.astype(np.float32), (1, 2, 0))

    img = img[:, :, ::-1]

    src = center_crop(src)

    return np.max(np.abs(src - img))


def run_model(model):
    """"""
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_mean = np.array(mean).reshape((3, 1, 1)).astype(np.float32)
    img_std = np.array(std).reshape((3, 1, 1)).astype(np.float32)

    loss = torch.nn.CrossEntropyLoss()

    lines = load_dev('dev.csv')
    for id, true, target in lines:
        true = int(true)
        target = int(target)

        image = process_image(cv2.imread(osp.join('images', id)).astype(np.float32))
        src_image = cv2.imread(osp.join('images', id)).astype(np.float32)

        cnt = 0
        # v = Variable(torch.from_numpy(image), requires_grad=True)
        # optimizer = torch.optim.SGD([v], lr=1e-4, momentum=0.9, weight_decay=0)
        while True:
            v = Variable(torch.from_numpy(image), requires_grad=True)
            optimizer = torch.optim.SGD([v], lr=1, momentum=0.9, weight_decay=0)

            # TODO: clip the image [-32, 32]
            optimizer.zero_grad()

            x = model(v.cuda())
            torch_x = x

            x = x.cpu().detach().numpy()
            pred = np.argmax(x) + 1

            inv_image = process_image_inv(v.cpu().detach().numpy(), src_image)
            dis = distance(src_image.copy(), v.cpu().detach().numpy().copy())

            print(dis)
            # if target == pred and cnt >= 1000:
            if cnt >= 1000:
                print(pred)
                cv2.imwrite(osp.join('fake_images', id), inv_image)
                break

            output = loss(torch_x, torch.from_numpy(np.array([target - 1])).cuda())  # need to use target - 1 to match the idx
            output.backward()
            optimizer.step()

            cnt += 1
            print(id, true, np.argmax(x) + 1, true == np.argmax(x) + 1, target, target == np.argmax(x) + 1, cnt)

            image = process_image(clip(src_image.copy(), inv_image))

        print()


if __name__ == '__main__':
    # resnet50_model = resnet50().eval().cuda()
    # resnet152_model = resnet152().eval().cuda()
    # vgg16_model = vgg16().eval().cuda()
    mobilenet_model = mobilenetv2().eval().cuda()

    run_model(mobilenet_model)
