#! /usr/bin/python
import os, sys
import util
from PIL import Image

trainval_dir = util.io.get_absolute_path("~/dataset_nfs/ICDAR2015/Challenge4/ch4_training_images")
test_dir = util.io.get_absolute_path("~/dataset_nfs/ICDAR2015/Challenge4/ch4_training_images")

trainval_img_lists = util.io.ls(trainval_dir, '.jpg')
trainval_img_names = []
for item in trainval_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    trainval_img_names.append(temp1)

test_img_lists = util.io.ls(test_dir, '.jpg')
test_img_names = []
for item in test_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_img_names.append(temp1)

dist_img_dir = "ch4_training_images"
dist_anno_dir = "ch4_training_localization_transcription_gt/xml"

trainval_fd = open("trainval.txt", 'w')
test_fd = open("test.txt", 'w')


for item in trainval_img_names:
    trainval_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')

for item in test_img_names:
    test_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')

