#! /usr/bin/python
import os, sys
import util
from PIL import Image

trainval_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task12_Images")
test_dir = util.io.get_absolute_path("~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task12_Images")

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

dist_img_dir = 'Challenge2_Training_Task12_Images'
dist_anno_dir = 'Challenge2_Training_Task1_GT/xml'
trainval_fd = open("trainval.txt", 'w')
for item in trainval_img_names:
    trainval_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')


dist_img_dir = 'Challenge2_Test_Task12_Images'
dist_anno_dir = 'Challenge2_Test_Task1_GT/xml'
test_fd = open("test.txt", 'w')
for item in test_img_names:
    test_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')

