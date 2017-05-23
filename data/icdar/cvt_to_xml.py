#! /usr/bin/python

import os, sys
import util
from PIL import Image

src_img_dir = util.io.get_absolute_path("~/dataset_nfs/ICDAR2015/Challenge4/ch4_training_images")
src_txt_dir = util.io.get_absolute_path("~/dataset_nfs/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt")

xml_dir = src_txt_dir + '/xml'
util.io.mkdir(xml_dir)
img_basenames = util.io.ls(src_img_dir, '.jpg')

img_names = [] # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)

for img in img_names:
    print img, '...'
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size

    # open the crospronding txt file
    gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()

    # write in xml file
    xml_file = open((xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>ICDAR2015</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of text on xml file
    for img_each_label in gt:
        img_each_label = util.str.remove_all(img_each_label, '\xef\xbb\xbf')
        spt = img_each_label.split(',')
        locs = spt[0: -1]
        xs = [int(locs[0]), int(locs[2]), int(locs[4]), int(locs[6])]
        ys = [int(locs[1]), int(locs[3]), int(locs[5]), int(locs[7])]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        xml_file.write('    <object>\n')
        xml_file.write('        <name>text</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')

