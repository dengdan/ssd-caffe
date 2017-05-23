#! /usr/bin/python

import os, sys
import glob
from PIL import Image
import util

img_dir = util.io.get_absolute_path("~/dataset_nfs/ICDAR2015/Challenge4/ch4_training_images")
img_lists = util.io.ls(img_dir)

test_name_size = open('test_name_size.txt', 'w')

for item in img_lists:
    img = Image.open(util.io.join_path(img_dir, item))
    width, height = img.size
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_name_size.write(temp1 + ' ' + str(height) + ' ' + str(width) + '\n')

