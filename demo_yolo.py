#!/usr/bin/env python
# ____cfy______
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import cv2
import numpy as np
import json
from db.datasets import datasets
from config import system_configs
from nnet.py_factory import NetworkFactory
from utils import  normalize_
from test.images import PostProcess
from sample.vis import *
from elements.yolo import YOLO
from elements.asset import plot_one_box
import random

torch.backends.cudnn.benchmark = False

detector = YOLO('weights/yolov5m.pt')
names = {
        'person': 0,
        'car' : 1,
        'bus': 2,
        'truck' : 3,
        'traffic light' : 4,
        'stop sign' : 5}
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

img_path = './test.jpg'
frame = cv2.imread(img_path)
main_frame = frame.copy()

hw = frame.shape
rotate = hw[1]<hw[0]
if rotate :
    hw[0],hw[1] = hw[1],hw[0]
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
need_resize = not((hw[1] == 1280) and (hw[0] == 720))
if need_resize:
    frame = cv2.resize(frame , (int(1280),int(720)))
print('resize:',need_resize)
print('rotate:',rotate)

yoloOutput = detector.detect(frame)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

result_dir = './demo_result'
cfg_file = './config/LSTR.json'
with open(cfg_file, "r") as f:
    configs = json.load(f)

configs["system"]["snapshot_name"] = 'LSTR'
system_configs.update_config(configs["system"])
split = system_configs.val_split

print("loading all datasets...")
dataset = system_configs.dataset
print("split: {}".format(split))  # test

testing_db = datasets[dataset](configs["db"], split)
test_iter = 800000
print("loading parameters at iteration: {}".format(test_iter))

print("building neural network...")
nnet = NetworkFactory()
print("loading parameters...")
nnet.load_params(test_iter)
nnet.cuda()
nnet.eval_mode()
print("模型载入成功！")


input_size  = [360, 640]
image = main_frame.copy()
height , width = image.shape[0:2]

images = np.zeros((1, 3, input_size[0], input_size[1]), dtype=np.float32)
masks = np.ones((1, 1, input_size[0], input_size[1]), dtype=np.float32)
orig_target_sizes = torch.tensor(input_size).unsqueeze(0).cuda()
pad_image     = image.copy()
pad_mask      = np.zeros((height, width, 1), dtype=np.float32)
resized_image = cv2.resize(pad_image, (input_size[1], input_size[0]))
resized_mask  = cv2.resize(pad_mask, (input_size[1], input_size[0]))
masks[0][0]   = resized_mask.squeeze()
resized_image = resized_image / 255.
normalize_(resized_image, IMAGENET_MEAN, IMAGENET_STD)
resized_image = resized_image.transpose(2, 0, 1)
images[0]     = resized_image
images        = torch.from_numpy(images).cuda(non_blocking=True)
masks         = torch.from_numpy(masks).cuda(non_blocking=True)
#torch.cuda.synchronize(0)   # 0 is the GPU id
outputs, weights      = nnet.test([images, masks])
#torch.cuda.synchronize(0)

postprocessors = {'bbox': PostProcess()}
results = postprocessors['bbox'](outputs, orig_target_sizes)
pred = results[0].cpu().numpy()
img  = pad_image
img_h, img_w, _ = img.shape
pred = pred[pred[:, 0].astype(int) == 1]
overlay = img.copy()
color = (0, 255, 0)

for i, lane in enumerate(pred):
    lane = lane[1:]  # remove conf
    lower, upper = lane[0], lane[1]
    lane = lane[2:]  # remove upper, lower positions

    # generate points from the polynomial
    ys = np.linspace(lower, upper, num=100)
    points = np.zeros((len(ys), 2), dtype=np.int32)
    points[:, 1] = (ys * img_h).astype(int)
    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys -
                        lane[5]) * img_w).astype(int)
    points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

    # draw lane with a polyline on the overlay
    for current_point, next_point in zip(points[:-1], points[1:]):
        overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=15)

    # draw lane ID
    if len(points) > 0:
        cv2.putText(img, str(i), tuple(points[0]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=color,
                    thickness=3)
# Add lanes overlay
w = 0.6
img = ((1. - w) * img + w * overlay).astype(np.uint8)
cv2.imwrite(os.path.join(result_dir,'lane.jpg') , img)
print('LSTR检测后的图像已保存！')

for obj in yoloOutput:
    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
    plot_one_box(xyxy, img, label=obj['label'], color=colors[names[obj['label']]], line_thickness=3)
cv2.imwrite(os.path.join(result_dir , 'yolo.jpg') , img)
print('YOLO检测后的图像已保存！')