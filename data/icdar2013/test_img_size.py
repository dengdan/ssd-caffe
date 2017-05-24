#! /usr/bin/python

import os, sys
import glob
from PIL import Image
import util

image_root = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/")
test_dir = "Challenge2_Test_Task12_Images"
train_dir = "Challenge2_Training_Task12_Images";
test_name_size = open('test_name_size.txt', 'w')

dirs = [test_dir, train_dir]
for d in dirs:
    path = util.io.join_path(image_root, d)
    images = util.io.ls(path, '.jpg')

    for item in images:
        img = Image.open(util.io.join_path(image_root, d, item))
        width, height = img.size
        name = util.io.get_filename(item).split('.')[0]
        test_name_size.write(util.io.join_path(d, name) + ' ' + str(height) + ' ' + str(width) + '\n')



