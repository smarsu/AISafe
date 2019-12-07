# Copyright (c) smarsu. All Rights Reserved.

import os.path as osp
import csv
import cv2
from tqdm import tqdm
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


from networks import resnet, mobilenet
import utils


def process(img, src_img):
  img = img.astype(np.float32)
  img = utils.clip(img, src_img)
  img = utils.crop(img)
  img = utils.norm(img)
  return img


def process_inv(img, src_img):
  img = img.astype(np.float32)
  img = utils.norm_inv(img)
  img = utils.crop_inv(img, src_img)
  # img = utils.clip(img, src_img)
  return img


def test(model, is_fake=True):
  lines = utils.load_dev('dev.csv')
  for id, true, target in tqdm(lines):
    # if id != '0c7ac4a8c9dfa802.png':
    #     continue
    if id != '62ebd5f7ce015380.png':
      continue

    root = 'fake_images' if is_fake else 'images'
    src_img = cv2.imread(osp.join(root, id))
    true = int(true)
    target = int(target)

    last_image = src_img.astype(np.float32)
    if True:
      processed_img = process(last_image, src_img.copy())
      pred = model(torch.from_numpy(processed_img).cuda())

      pred_cpu = pred.cpu().detach().numpy()
      pred_target = np.argmax(pred_cpu) + 1

      print('id:', id,
            'true:', true, 
            'pred:', pred_target, 
            'target:', target)


def run_model(model):
  """Train the fake model."""
  loss = nn.MSELoss()
  # loss = nn.CrossEntropyLoss()

  lines = utils.load_dev('dev.csv')
  for id, true, target in tqdm(lines):
    src_img = cv2.imread(osp.join('images', id))
    true = int(true)
    target = int(target)

    last_image = src_img.astype(np.float32)
    cnt = 0
    # processed_img = process(last_image, src_img.copy())
    # v = Variable(torch.from_numpy(processed_img), requires_grad=True)
    # optimizer = torch.optim.SGD([v], lr=1, momentum=0.9, weight_decay=0)
    while True:
      processed_img = process(last_image, src_img.copy())
      v = Variable(torch.from_numpy(processed_img), requires_grad=True)
      optimizer = torch.optim.SGD([v], lr=100, momentum=0.9, weight_decay=0)

      optimizer.zero_grad()
      pred = model(v.cuda())

      pred_cpu = pred.cpu().detach().numpy()
      pred_target = np.argmax(pred_cpu) + 1

      # inv_img = process_inv(v.cpu().detach().numpy().copy(), src_img.copy())
      dis = utils.distance(src_img.copy(), last_image.copy())

      if cnt >= 1000:
        print('id:', id,
              'true:', true, 
              'pred:', pred_target, 
              'target:', target, 
              'dis:', dis,
              'cnt:', cnt)
        cv2.imwrite(osp.join('fake_images', id), last_image)
        break

      # output = loss(pred, torch.from_numpy(np.array([target - 1])).cuda())
      # output = loss(pred, torch.from_numpy(np.array(utils.onehot(target - 1), dtype=np.float32)).cuda())
      output = loss(pred, torch.from_numpy(np.array(utils.onehot_reverse(true - 1), dtype=np.float32)).cuda())
      output.backward()
      optimizer.step()

      last_image = process_inv(v.cpu().detach().numpy().copy(), src_img.copy())

      cnt += 1
      print('id:', id,
            'true:', true, 
            'pred:', pred_target, 
            'target:', target, 
            'dis:', dis,
            'cnt:', cnt)


if __name__ == '__main__':
  models = [resnet.resnet50(pretrained=True, progress=True).eval().cuda(),
            mobilenet.mobilenet_v2(pretrained=True, progress=True).eval().cuda(),
            resnet.resnet152(pretrained=True, progress=True).eval().cuda(),]
            
  model = resnet.resnet50(pretrained=True, progress=True).eval().cuda()

  # run_model(model)
  [test(model, is_fake=True) for model in models] 
