# Copyright (c) smarsu. All Rights Reserved.

import numpy as np
import csv
import cv2

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
img_mean = np.array(mean).reshape((3, 1, 1))
img_std = np.array(std).reshape((3, 1, 1))

def crop(image):
  image = image.astype(np.float32)

  assert image.shape == (299, 299, 3)
  image = cv2.resize(image, (256, 256))
  h, w, _ = image.shape
  top = (h - 224) // 2
  bottom = (h - 224) - top
  left = (w - 224) // 2
  right = (w - 224) - left

  image = image[top:h-bottom, left:w-right, :]
  assert image.shape == (224, 224, 3)
  return image


def crop_inv(image, src_image):
  image = image.astype(np.float32)
  src_image = src_image.astype(np.float32)

  assert image.shape == (224, 224, 3)
  assert src_image.shape == (299, 299, 3)

  src_image = cv2.resize(src_image, (256, 256))
  h, w, _ = src_image.shape
  top = (h - 224) // 2
  bottom = (h - 224) - top
  left = (w - 224) // 2
  right = (w - 224) - left

  src_image[top:h-bottom, left:w-right, :] = image
  src_image = cv2.resize(src_image, (299, 299))
  return src_image


# def crop(image):
#   image = image.astype(np.float32)

#   assert image.shape == (299, 299, 3)
#   # image = cv2.resize(image, (256, 256))
#   h, w, _ = image.shape
#   top = (h - 224) // 2
#   bottom = (h - 224) - top
#   left = (w - 224) // 2
#   right = (w - 224) - left

#   image = image[top:h-bottom, left:w-right, :]
#   assert image.shape == (224, 224, 3)
#   return image


# def crop_inv(image, src_image):
#   image = image.astype(np.float32)
#   src_image = src_image.astype(np.float32)

#   assert image.shape == (224, 224, 3)
#   assert src_image.shape == (299, 299, 3)

#   h, w, _ = src_image.shape
#   top = (h - 224) // 2
#   bottom = (h - 224) - top
#   left = (w - 224) // 2
#   right = (w - 224) - left

#   src_image[top:h-bottom, left:w-right, :] = image
#   return src_image


def norm(img):
  """norm from gbr"""
  assert len(img.shape) == 3
  img = img[:, :, ::-1]
  img = img.astype(np.float32).transpose(2, 0, 1)

  img /= 255
  img -= img_mean
  img /= img_std

  return img[np.newaxis, ...]


def norm_inv(img):
  assert len(img.shape) == 4
  img = img.astype(np.float32)
  img = img[0]

  img *= img_std
  img += img_mean
  img *= 255

  img = img.transpose(1, 2, 0)
  img = img[:, :, ::-1]
  return img


def clip(dst_img, img, border=32):
  dst_img = dst_img.astype(np.float32)
  img = img.astype(np.float32)

  left = img - border
  left = np.clip(left, 0, 255)

  right = img + border
  right = np.clip(right, 0, 255)

  dst_img = np.clip(dst_img, left, right)
  return dst_img


def distance(x, y):
  x = x.astype(np.float32)
  y = y.astype(np.float32)
  return np.max(np.abs(x - y))


def onehot(x, num=1000):
  dst = [0 for _ in range(num)]
  dst[x] = 1
  return dst


def onehot_reverse(x, num=1000):
  dst = [1 for _ in range(num)]
  dst[x] = 0
  return dst


def load_dev(path):
    """"""
    with open(path, "r") as fb:
        reader = list(csv.reader(fb))[1:]  # no need for title
        return reader
